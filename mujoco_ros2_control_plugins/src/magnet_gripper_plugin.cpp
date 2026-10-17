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

#include "mujoco_ros2_control_plugins/magnet_gripper_plugin.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <pluginlib/class_list_macros.hpp>

namespace mujoco_ros2_control_plugins
{

namespace
{
// Sub-nodes share the parent's parameters without a prefix; this plugin's settings live under
// mujoco_plugins.<plugin name>.<param> (same as JointCommandPlugin).
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

bool is_piece(const mjModel* m, int g)
{
  return m->geom_type[g] == mjGEOM_MESH;
}
}  // namespace

bool MagnetGripperPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* m, mjData* data)
{
  phys_data_ = data;   // the physics thread's mjData (init receives it; update() gets the control copy)
  node_ = node;
  logger_ = node_->get_logger();

  const auto gripper = declare_or_get<std::string>(node_, "gripper_body", "");
  const auto cells = declare_or_get<std::vector<std::string>>(node_, "cell_geoms", {});
  const auto prefix = declare_or_get<std::string>(node_, "plate_prefix", "inbound/");
  attach_distance_ = declare_or_get<double>(node_, "attach_distance", 0.01);
  released_piece_ = declare_or_get<std::vector<int64_t>>(node_, "released_piece", { 1, 15 });
  released_sphere_ = declare_or_get<std::vector<int64_t>>(node_, "released_sphere", { 8, 0 });
  frozen_zone_ = declare_or_get<std::vector<double>>(node_, "frozen_zone", std::vector<double>{});
  frozen_bits_ = declare_or_get<std::vector<int64_t>>(node_, "frozen_bits", { 32, 0 });
  if (!frozen_zone_.empty() && frozen_zone_.size() != 4)
  {
    RCLCPP_ERROR(logger_, "MagnetGripperPlugin: frozen_zone needs [xmin, xmax, ymin, ymax].");
    return false;
  }
  const auto on_name = declare_or_get<std::string>(node_, "on_service", "magnet_on");
  const auto off_name = declare_or_get<std::string>(node_, "off_service", "magnet_off");
  const auto state_name = declare_or_get<std::string>(node_, "state_topic", "magnet_state");
  const auto tf_prefix = declare_or_get<std::string>(node_, "tf_prefix", "plate/");
  const double tf_rate = declare_or_get<double>(node_, "tf_rate", 10.0);
  tf_period_ = tf_rate > 0 ? 1.0 / tf_rate : 0.0;

  gripper_body_ = mj_name2id(m, mjOBJ_BODY, gripper.c_str());
  if (gripper_body_ < 0)
  {
    RCLCPP_ERROR(logger_, "MagnetGripperPlugin: gripper_body '%s' not found.", gripper.c_str());
    return false;
  }
  for (const auto& c : cells)
  {
    const int g = mj_name2id(m, mjOBJ_GEOM, c.c_str());
    if (g < 0)
    {
      RCLCPP_ERROR(logger_, "MagnetGripperPlugin: cell geom '%s' not found.", c.c_str());
      return false;
    }
    cell_geoms_.push_back(g);
  }
  if (cell_geoms_.empty())
  {
    RCLCPP_ERROR(logger_, "MagnetGripperPlugin: 'cell_geoms' is empty.");
    return false;
  }
  for (int b = 1; b < m->nbody; ++b)
  {
    const char* n = mj_id2name(m, mjOBJ_BODY, b);
    if (n && std::string(n).rfind(prefix, 0) == 0 && m->body_jntnum[b] == 1 &&
        m->jnt_type[m->body_jntadr[b]] == mjJNT_FREE)
    {
      const std::string eq_name = "magnet/" + std::string(n);
      const int eq = mj_name2id(m, mjOBJ_EQUALITY, eq_name.c_str());
      if (eq < 0 || m->eq_type[eq] != mjEQ_WELD)
      {
        RCLCPP_ERROR(logger_, "MagnetGripperPlugin: weld equality '%s' missing (build_rc_cell_mjcf.py adds it).",
                     eq_name.c_str());
        return false;
      }
      plate_bodies_.push_back(b);
      plate_eq_.push_back(eq);
    }
  }

  on_srv_ = node_->create_service<std_srvs::srv::Trigger>(
      on_name, [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        std::tie(res->success, res->message) = call(true);
      });
  off_srv_ = node_->create_service<std_srvs::srv::Trigger>(
      off_name, [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        std::tie(res->success, res->message) = call(false);
      });
  state_pub_ = node_->create_publisher<std_msgs::msg::String>(state_name, rclcpp::QoS(1).transient_local());
  scratch_ = mj_makeData(m);
  tf_pub_ = node_->create_publisher<tf2_msgs::msg::TFMessage>("/tf", 10);
  for (int b : plate_bodies_)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.frame_id = "world";
    t.child_frame_id = tf_prefix + std::string(mj_id2name(m, mjOBJ_BODY, b)).substr(prefix.size());
    tf_msg_.transforms.push_back(t);
  }

  RCLCPP_INFO(logger_, "MagnetGripperPlugin: gripper '%s', %zu cells, %zu plates ('%s*'). on=%s off=%s state=%s",
              gripper.c_str(), cell_geoms_.size(), plate_bodies_.size(), prefix.c_str(), on_srv_->get_service_name(),
              off_srv_->get_service_name(), state_pub_->get_topic_name());
  return true;
}

std::pair<bool, std::string> MagnetGripperPlugin::call(bool on)
{
  std::future<std::pair<bool, std::string>> fut;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_)
    {
      return { false, "previous magnet request still pending" };
    }
    pending_.emplace(Request{ on, {} });
    fut = pending_->result.get_future();
  }
  if (fut.wait_for(std::chrono::seconds(5)) != std::future_status::ready)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.reset();
    return { false, "timeout: simulation not stepping?" };
  }
  return fut.get();
}

void MagnetGripperPlugin::set_plate_contacts(const mjModel* m, int body, int piece_ct, int piece_ca, int sph_ct,
                                              int sph_ca)
{
  // Model fields are shared with the physics thread; the caller holds the simulation mutex (see update()).
  auto* mm = const_cast<mjModel*>(m);
  int body_ct = 0, body_ca = 0;
  for (int g = 0; g < m->ngeom; ++g)
  {
    // group 1 = plate visual mesh (build_rc_cell_mjcf.py add_plate_visual): never collides — it has no convex hull.
    if (m->geom_bodyid[g] != body || m->geom_group[g] == 1)
    {
      continue;
    }
    const bool piece = is_piece(m, g);
    mm->geom_contype[g] = piece ? piece_ct : sph_ct;
    mm->geom_conaffinity[g] = piece ? piece_ca : sph_ca;
    body_ct |= mm->geom_contype[g];
    body_ca |= mm->geom_conaffinity[g];
  }
  // The broadphase filters on the per-body OR first — without this the released plate fell through everything.
  mm->body_contype[body] = body_ct;
  mm->body_conaffinity[body] = body_ca;
}

std::pair<bool, std::string> MagnetGripperPlugin::attach(const mjModel* m, mjData* d)
{
  if (carried_ >= 0)
  {
    return { false, std::string("already holding ") + mj_id2name(m, mjOBJ_BODY, carried_) };
  }
  // Measure on a private mjData: mj_ray / mj_kinematics must not touch the physics data's state or arena.
  mju_copy(scratch_->qpos, d->qpos, m->nq);
  mj_kinematics(m, scratch_);
  // Each cell shoots a ray out of its face along the tool axis (gripper body +X); the first thing it hits must be a
  // plate within attach_distance. Closest-geometry distance also caught a neighbouring plate's raised edge next to
  // the face (tilted ga1_03 beside ga2_04) and welded that plate at a large offset. The plate under the most cells
  // wins, ties broken by the smallest gap.
  const mjtNum* gx = scratch_->xmat + 9 * gripper_body_;
  const mjtNum dir[3] = { gx[0], gx[3], gx[6] };
  std::vector<int> n_cells(m->nbody, 0);
  std::vector<mjtNum> gap(m->nbody, std::numeric_limits<mjtNum>::infinity());
  for (int c : cell_geoms_)
  {
    // centre -> face distance along dir (box half extents projected on dir)
    const mjtNum* cx = scratch_->geom_xmat + 9 * c;
    mjtNum half = 0;
    for (int k = 0; k < 3; ++k)
    {
      half += m->geom_size[3 * c + k] * std::abs(cx[k] * dir[0] + cx[3 + k] * dir[1] + cx[6 + k] * dir[2]);
    }
    int hit = -1;
    const mjtNum dist = mj_ray(m, scratch_, scratch_->geom_xpos + 3 * c, dir, nullptr, 1, gripper_body_, &hit);
    if (hit < 0 || dist < 0 || dist - half > attach_distance_)
    {
      continue;
    }
    const int b = m->geom_bodyid[hit];
    if (std::find(plate_bodies_.begin(), plate_bodies_.end(), b) == plate_bodies_.end())
    {
      continue;
    }
    ++n_cells[b];
    gap[b] = std::min(gap[b], dist - half);
  }
  int best = -1;
  for (int b : plate_bodies_)
  {
    if (n_cells[b] > 0 &&
        (best < 0 || n_cells[b] > n_cells[best] || (n_cells[b] == n_cells[best] && gap[b] < gap[best])))
    {
      best = b;
    }
  }
  if (best < 0)
  {
    return { false, "no plate within " + std::to_string(attach_distance_ * 1e3) + " mm in front of the magnet cells" };
  }
  const int best_cells = n_cells[best];
  const mjtNum best_dist = gap[best];

  // Weld gripper -> plate at the current relative pose. Weld data: anchor(3) on body2, relpos(3) = anchor position
  // in body1 frame, relquat(4) = body2 orientation in body1 frame, torquescale. d is the physics mjData and the
  // simulation mutex is held, so eq_data / eq_active change between steps, never inside one.
  const int eq = plate_eq_[std::find(plate_bodies_.begin(), plate_bodies_.end(), best) - plate_bodies_.begin()];
  auto* mm = const_cast<mjModel*>(m);
  mjtNum* ed = mm->eq_data + mjNEQDATA * eq;
  mjtNum dp[3], q1[4], q2[4], q1i[4];
  mju_sub3(dp, d->xpos + 3 * best, d->xpos + 3 * gripper_body_);
  mju_zero3(ed);
  mju_mulMatTVec3(ed + 3, d->xmat + 9 * gripper_body_, dp);
  mju_copy4(q1, d->xquat + 4 * gripper_body_);
  mju_copy4(q2, d->xquat + 4 * best);
  mju_negQuat(q1i, q1);
  mju_mulQuat(ed + 6, q1i, q2);
  ed[10] = 1.0;

  // Weightless and contact-free while carried; nothing else acts on it.
  mm->body_gravcomp[best] = 1.0;
  set_plate_contacts(m, best, 0, 0, 0, 0);
  carried_eq_ = eq;
  phys_data_->eq_active[eq] = 1;
  carried_ = best;
  const std::string name = mj_id2name(m, mjOBJ_BODY, best);
  RCLCPP_INFO(logger_, "magnet ON: %s (%d/%zu cells, gap %.1f mm)", name.c_str(), best_cells, cell_geoms_.size(),
              std::max<mjtNum>(best_dist, 0) * 1e3);
  return { true, name };
}

std::pair<bool, std::string> MagnetGripperPlugin::release(const mjModel* m)
{
  if (carried_ < 0)
  {
    return { true, "nothing held" };
  }
  const std::string name = mj_id2name(m, mjOBJ_BODY, carried_);
  phys_data_->eq_active[carried_eq_] = 0;
  carried_eq_ = -1;
  const mjtNum* p = phys_data_->xpos + 3 * carried_;
  const bool frozen = frozen_zone_.size() == 4 && p[0] >= frozen_zone_[0] && p[0] <= frozen_zone_[1] &&
                      p[1] >= frozen_zone_[2] && p[1] <= frozen_zone_[3];
  if (frozen)
  {
    // put back on the static inbound stack: stays where it is, like the other stack plates
    const_cast<mjModel*>(m)->body_gravcomp[carried_] = 1.0;
    const int fct = static_cast<int>(frozen_bits_.at(0)), fca = static_cast<int>(frozen_bits_.at(1));
    set_plate_contacts(m, carried_, fct, fca, fct, fca);
    mju_zero(phys_data_->qvel + m->jnt_dofadr[m->body_jntadr[carried_]], 6);
  }
  else
  {
    const_cast<mjModel*>(m)->body_gravcomp[carried_] = 0.0;
    set_plate_contacts(m, carried_, static_cast<int>(released_piece_.at(0)), static_cast<int>(released_piece_.at(1)),
                       static_cast<int>(released_sphere_.at(0)), static_cast<int>(released_sphere_.at(1)));
  }
  carried_ = -1;
  RCLCPP_INFO(logger_, "magnet OFF: released %s%s", name.c_str(), frozen ? " (frozen zone: kept static on the stack)" : "");
  return { true, name };
}

void MagnetGripperPlugin::update(const mjModel* m, mjData* d)
{
  {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (lock.owns_lock() && pending_)
    {
      // Attach/release toggle eq_active and change model fields that mj_step reads. Doing that while the physics
      // thread was mid-step killed the sim ("mj_makeConstraint: ne mis-allocation: found ne=6 but allocated 0"), so
      // hold the simulation mutex (the physics thread holds it while stepping) and work on the physics mjData.
      std::unique_lock<std::recursive_mutex> sim_lock;
      if (sim_mutex_)
      {
        sim_lock = std::unique_lock<std::recursive_mutex>(*sim_mutex_);
      }
      mjData* pd = sim_mutex_ ? phys_data_ : d;
      pending_->result.set_value(pending_->on ? attach(m, pd) : release(m));
      pending_.reset();
    }
  }

  if (tf_period_ > 0 && (d->time - last_tf_pub_ >= tf_period_ || d->time < last_tf_pub_))
  {
    last_tf_pub_ = d->time;
    const auto stamp = node_->get_clock()->now();
    for (std::size_t i = 0; i < plate_bodies_.size(); ++i)
    {
      const int b = plate_bodies_[i];
      auto& t = tf_msg_.transforms[i];
      t.header.stamp = stamp;
      t.transform.translation.x = d->xpos[3 * b];
      t.transform.translation.y = d->xpos[3 * b + 1];
      t.transform.translation.z = d->xpos[3 * b + 2];
      t.transform.rotation.w = d->xquat[4 * b];
      t.transform.rotation.x = d->xquat[4 * b + 1];
      t.transform.rotation.y = d->xquat[4 * b + 2];
      t.transform.rotation.z = d->xquat[4 * b + 3];
    }
    tf_pub_->publish(tf_msg_);
  }

  if (d->time - last_state_pub_ >= 0.1 || d->time < last_state_pub_)
  {
    last_state_pub_ = d->time;
    std_msgs::msg::String msg;
    msg.data = carried_ >= 0 ? mj_id2name(m, mjOBJ_BODY, carried_) : "";
    state_pub_->publish(msg);
  }
}

void MagnetGripperPlugin::cleanup()
{
  if (scratch_)
  {
    mj_deleteData(scratch_);
    scratch_ = nullptr;
  }
  on_srv_.reset();
  off_srv_.reset();
  state_pub_.reset();
  tf_pub_.reset();
  node_.reset();
}

}  // namespace mujoco_ros2_control_plugins

PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_plugins::MagnetGripperPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
