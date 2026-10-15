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

#ifndef MUJOCO_ROS2_CONTROL_PLUGINS__JOINT_COMMAND_PLUGIN_HPP_
#define MUJOCO_ROS2_CONTROL_PLUGINS__JOINT_COMMAND_PLUGIN_HPP_

#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace mujoco_ros2_control_plugins
{

/**
 * @brief Drives MuJoCo position actuators directly, outside the controller manager.
 *
 * For a robot that shares the MJCF world with a ros2_control robot but is not itself under
 * ros2_control (its joints are left out of the <ros2_control> block, so MujocoSystemInterface
 * does not write their ctrl). Typical use: a vendor-controlled robot whose real interface is a
 * pose/joint move command, so the sim only needs to put the arm where it is told.
 *
 * Parameters (under mujoco_plugins.<name>.; use absolute topic names — relative ones resolve under the
 * hardware node's namespace + <name>):
 *   joints            MuJoCo joint names to drive (each needs a position actuator)
 *   output_names      names used on the ROS side, same order as joints (default: joints)
 *   state_topic       sensor_msgs/JointState output             (default: "joint_states")
 *   command_topic     sensor_msgs/JointState position targets   (default: "joint_command")
 *   trajectory_topic  trajectory_msgs/JointTrajectory to follow (default: "joint_trajectory")
 *   publish_rate      state publish rate [Hz]                   (default: 100)
 *   max_velocity      speed limit when moving to a joint_command target [rad/s or m/s] (default: 0.5)
 *
 * joint_command moves every named joint toward its target at max_velocity (partial name lists
 * are fine). joint_trajectory is played back in simulation time with linear interpolation,
 * starting from the current command; a new message replaces whatever is running.
 */
class JointCommandPlugin : public MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override;
  void update(const mjModel* model, mjData* data) override;
  void cleanup() override;

private:
  void on_command(const sensor_msgs::msg::JointState::SharedPtr msg);
  void on_trajectory(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);
  int index_of(const std::string& name) const;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("JointCommandPlugin") };

  std::vector<std::string> joint_names_;
  std::vector<std::string> output_names_;
  std::vector<int> qpos_adr_;
  std::vector<int> qvel_adr_;
  std::vector<int> actuator_id_;

  // Guarded by mutex_: written by subscription callbacks, read by update().
  std::mutex mutex_;
  std::vector<double> target_;
  bool has_new_trajectory_{ false };
  trajectory_msgs::msg::JointTrajectory pending_trajectory_;
  std::vector<int> pending_trajectory_map_;

  // Only touched in update().
  bool initialized_{ false };
  std::vector<double> command_;
  std::vector<double> step_target_;
  double last_time_{ 0.0 };
  bool trajectory_active_{ false };
  double trajectory_start_time_{ 0.0 };
  std::vector<double> trajectory_start_;
  trajectory_msgs::msg::JointTrajectory trajectory_;
  std::vector<int> trajectory_map_;

  double max_velocity_{ 0.5 };
  double publish_period_{ 0.01 };
  double last_publish_time_{ -1.0 };

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr command_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_sub_;
  sensor_msgs::msg::JointState state_msg_;
};

}  // namespace mujoco_ros2_control_plugins

#endif  // MUJOCO_ROS2_CONTROL_PLUGINS__JOINT_COMMAND_PLUGIN_HPP_
