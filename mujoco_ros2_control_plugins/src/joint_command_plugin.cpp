// Copyright 2026 Aidin Robotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mujoco_ros2_control_plugins/joint_command_plugin.hpp"

#include <algorithm>
#include <cmath>

#include <pluginlib/class_list_macros.hpp>

namespace mujoco_ros2_control_plugins
{

namespace
{
// Sub-nodes share the parent node's parameters without a prefix, and the yaml puts this plugin's
// settings under mujoco_plugins.<plugin name>.<param>, so look them up by the full name.
template <typename T>
T declare_or_get(const rclcpp::Node::SharedPtr& node, const std::string& name, const T& default_value)
{
  const std::string full = "mujoco_plugins." + node->get_sub_namespace() + "." + name;
  if (!node->has_parameter(full))
  {
    node->declare_parameter(full, default_value);
  }
  return node->get_parameter(full).get_value<T>();
}

double to_seconds(const builtin_interfaces::msg::Duration& d)
{
  return static_cast<double>(d.sec) + static_cast<double>(d.nanosec) * 1e-9;
}
}  // namespace

bool JointCommandPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* /*data*/)
{
  node_ = node;
  logger_ = node_->get_logger();

  joint_names_ = declare_or_get<std::vector<std::string>>(node_, "joints", {});
  output_names_ = declare_or_get<std::vector<std::string>>(node_, "output_names", {});
  const auto state_topic = declare_or_get<std::string>(node_, "state_topic", "joint_states");
  const auto command_topic = declare_or_get<std::string>(node_, "command_topic", "joint_command");
  const auto trajectory_topic = declare_or_get<std::string>(node_, "trajectory_topic", "joint_trajectory");
  const double publish_rate = declare_or_get<double>(node_, "publish_rate", 100.0);
  max_velocity_ = declare_or_get<double>(node_, "max_velocity", 0.5);

  if (joint_names_.empty())
  {
    RCLCPP_ERROR(logger_, "JointCommandPlugin: parameter 'joints' is empty.");
    return false;
  }
  if (output_names_.empty())
  {
    output_names_ = joint_names_;
  }
  if (output_names_.size() != joint_names_.size())
  {
    RCLCPP_ERROR(logger_, "JointCommandPlugin: 'output_names' (%zu) and 'joints' (%zu) differ in size.",
                 output_names_.size(), joint_names_.size());
    return false;
  }
  publish_period_ = publish_rate > 0.0 ? 1.0 / publish_rate : 0.0;

  for (const auto& name : joint_names_)
  {
    const int jid = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    if (jid < 0)
    {
      RCLCPP_ERROR(logger_, "JointCommandPlugin: joint '%s' not found in the MJCF.", name.c_str());
      return false;
    }
    int act = -1;
    for (int i = 0; i < model->nu; ++i)
    {
      if (model->actuator_trntype[i] == mjTRN_JOINT && model->actuator_trnid[2 * i] == jid)
      {
        act = i;
        break;
      }
    }
    if (act < 0)
    {
      RCLCPP_ERROR(logger_, "JointCommandPlugin: joint '%s' has no actuator.", name.c_str());
      return false;
    }
    qpos_adr_.push_back(model->jnt_qposadr[jid]);
    qvel_adr_.push_back(model->jnt_dofadr[jid]);
    actuator_id_.push_back(act);
  }

  const std::size_t n = joint_names_.size();
  target_.assign(n, 0.0);
  command_.assign(n, 0.0);
  step_target_.assign(n, 0.0);

  state_msg_.name = output_names_;
  state_msg_.position.assign(n, 0.0);
  state_msg_.velocity.assign(n, 0.0);
  state_msg_.effort.assign(n, 0.0);

  state_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(state_topic, 10);
  command_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      command_topic, 10, [this](const sensor_msgs::msg::JointState::SharedPtr msg) { on_command(msg); });
  trajectory_sub_ = node_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      trajectory_topic, 10,
      [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) { on_trajectory(msg); });

  RCLCPP_INFO(logger_, "JointCommandPlugin: driving %zu joints. state=%s command=%s trajectory=%s", n,
              state_pub_->get_topic_name(), command_sub_->get_topic_name(), trajectory_sub_->get_topic_name());
  return true;
}

int JointCommandPlugin::index_of(const std::string& name) const
{
  const auto it = std::find(output_names_.begin(), output_names_.end(), name);
  return it == output_names_.end() ? -1 : static_cast<int>(it - output_names_.begin());
}

void JointCommandPlugin::on_command(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (msg->position.size() != msg->name.size())
  {
    RCLCPP_WARN(logger_, "joint_command: name/position size mismatch, ignored.");
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (std::size_t i = 0; i < msg->name.size(); ++i)
  {
    const int idx = index_of(msg->name[i]);
    if (idx < 0)
    {
      RCLCPP_WARN(logger_, "joint_command: unknown joint '%s', ignored.", msg->name[i].c_str());
      continue;
    }
    target_[idx] = msg->position[i];
  }
  // A step target cancels any running trajectory.
  has_new_trajectory_ = true;
  pending_trajectory_ = trajectory_msgs::msg::JointTrajectory();
  pending_trajectory_map_.clear();
}

void JointCommandPlugin::on_trajectory(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  std::vector<int> map;
  for (const auto& name : msg->joint_names)
  {
    const int idx = index_of(name);
    if (idx < 0)
    {
      RCLCPP_WARN(logger_, "joint_trajectory: unknown joint '%s', trajectory rejected.", name.c_str());
      return;
    }
    map.push_back(idx);
  }
  for (const auto& p : msg->points)
  {
    if (p.positions.size() != map.size())
    {
      RCLCPP_WARN(logger_, "joint_trajectory: point size mismatch, trajectory rejected.");
      return;
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  has_new_trajectory_ = true;
  pending_trajectory_ = *msg;
  pending_trajectory_map_ = map;
}

void JointCommandPlugin::update(const mjModel* /*model*/, mjData* data)
{
  const std::size_t n = joint_names_.size();
  if (!initialized_)
  {
    // First cycle: start from where the arm actually is (initial keyframe already applied).
    for (std::size_t i = 0; i < n; ++i)
    {
      command_[i] = data->qpos[qpos_adr_[i]];
    }
    step_target_ = command_;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      target_ = command_;
    }
    last_time_ = data->time;
    initialized_ = true;
  }

  const double now = data->time;
  const double dt = std::max(0.0, now - last_time_);
  last_time_ = now;

  // Pick up new requests without blocking the real-time loop.
  {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (lock.owns_lock())
    {
      step_target_ = target_;
      if (has_new_trajectory_)
      {
        has_new_trajectory_ = false;
        trajectory_active_ = !pending_trajectory_.points.empty();
        if (trajectory_active_)
        {
          trajectory_ = pending_trajectory_;
          trajectory_map_ = pending_trajectory_map_;
          trajectory_start_time_ = now;
          trajectory_start_ = command_;
        }
      }
    }
  }

  if (trajectory_active_)
  {
    const double t = now - trajectory_start_time_;
    const auto& pts = trajectory_.points;
    // Segment [prev, next] containing t; before the first point interpolate from the start command.
    std::size_t k = 0;
    while (k < pts.size() && to_seconds(pts[k].time_from_start) < t)
    {
      ++k;
    }
    if (k >= pts.size())
    {
      for (std::size_t j = 0; j < trajectory_map_.size(); ++j)
      {
        command_[trajectory_map_[j]] = pts.back().positions[j];
      }
      trajectory_active_ = false;
      // Hold the final point afterwards.
      std::lock_guard<std::mutex> lock(mutex_);
      target_ = command_;
      step_target_ = command_;
    }
    else
    {
      const double t1 = to_seconds(pts[k].time_from_start);
      const double t0 = k == 0 ? 0.0 : to_seconds(pts[k - 1].time_from_start);
      const double a = t1 > t0 ? std::clamp((t - t0) / (t1 - t0), 0.0, 1.0) : 1.0;
      for (std::size_t j = 0; j < trajectory_map_.size(); ++j)
      {
        const int idx = trajectory_map_[j];
        const double p0 = k == 0 ? trajectory_start_[idx] : pts[k - 1].positions[j];
        command_[idx] = p0 + a * (pts[k].positions[j] - p0);
      }
    }
  }
  else
  {
    const double max_step = max_velocity_ * dt;
    for (std::size_t i = 0; i < n; ++i)
    {
      command_[i] += std::clamp(step_target_[i] - command_[i], -max_step, max_step);
    }
  }

  for (std::size_t i = 0; i < n; ++i)
  {
    data->ctrl[actuator_id_[i]] = command_[i];
  }

  if (publish_period_ <= 0.0 || last_publish_time_ < 0.0 || now - last_publish_time_ >= publish_period_ ||
      now < last_publish_time_)
  {
    last_publish_time_ = now;
    state_msg_.header.stamp = node_->get_clock()->now();
    for (std::size_t i = 0; i < n; ++i)
    {
      state_msg_.position[i] = data->qpos[qpos_adr_[i]];
      state_msg_.velocity[i] = data->qvel[qvel_adr_[i]];
      state_msg_.effort[i] = data->actuator_force[actuator_id_[i]];
    }
    state_pub_->publish(state_msg_);
  }
}

void JointCommandPlugin::cleanup()
{
  trajectory_sub_.reset();
  command_sub_.reset();
  state_pub_.reset();
  node_.reset();
}

}  // namespace mujoco_ros2_control_plugins

PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_plugins::JointCommandPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
