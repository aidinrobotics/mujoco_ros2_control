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

#include "mujoco_ros2_control_plugins/table_magnet_plugin.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>

#include <pluginlib/class_list_macros.hpp>

namespace mujoco_ros2_control_plugins
{

namespace
{
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

std::atomic<int> g_parked{ 0 };  // plates parked by clear_service (all tables) — next park slot

// roll/pitch removed: the rotation that takes the body +Z onto world +Z, applied in front of q
void level_quat(const mjtNum* xmat, const mjtNum* q, mjtNum* out)
{
  const mjtNum zb[3] = { xmat[2], xmat[5], xmat[8] };
  const mjtNum ez[3] = { 0, 0, 1 };
  mjtNum axis[3];
  mju_cross(axis, zb, ez);
  const mjtNum s = mju_norm3(axis);
  const mjtNum c = mju_dot3(zb, ez);
  if (s < 1e-12)
  {
    mju_copy4(out, q);
    return;
  }
  mju_scl3(axis, axis, 1.0 / s);
  mjtNum qfix[4];
  mju_axisAngle2Quat(qfix, axis, std::atan2(s, c));
  mju_mulQuat(out, qfix, q);
  mju_normalize4(out);
}

double tilt_deg(const mjtNum* xmat)
{
  return std::acos(std::clamp<double>(xmat[8], -1.0, 1.0)) * 180.0 / M_PI;
}
}  // namespace

bool TableMagnetPlugin::init(rclcpp::Node::SharedPtr node, const mjModel* m, mjData* data)
{
  phys_data_ = data;
  node_ = node;
  logger_ = node_->get_logger();

  const auto posts = declare_or_get<std::vector<std::string>>(node_, "post_bodies", {});
  post_radius_ = declare_or_get<double>(node_, "post_radius", 0.05);
  const auto prefix = declare_or_get<std::string>(node_, "plate_prefix", "inbound/");
  attach_distance_ = declare_or_get<double>(node_, "attach_distance", 0.03);
  settle_ = declare_or_get<bool>(node_, "settle", true);
  settle_clearance_ = declare_or_get<double>(node_, "settle_clearance", 0.0005);
  const auto grip_prefix = declare_or_get<std::string>(node_, "gripper_weld_prefix", "magnet/");
  const auto on_name = declare_or_get<std::string>(node_, "on_service", "magnet_on");
  const auto off_name = declare_or_get<std::string>(node_, "off_service", "magnet_off");
  const auto clear_name = declare_or_get<std::string>(node_, "clear_service", "");
  const auto state_name = declare_or_get<std::string>(node_, "state_topic", "magnet_state");
  park_position_ = declare_or_get<std::vector<double>>(node_, "park_position", park_position_);
  park_step_ = declare_or_get<std::vector<double>>(node_, "park_step", park_step_);
  if (park_position_.size() != 3 || park_step_.size() != 3)
  {
    RCLCPP_ERROR(logger_, "TableMagnetPlugin: park_position / park_step need 3 values.");
    return false;
  }

  for (const auto& p : posts)
  {
    const int b = mj_name2id(m, mjOBJ_BODY, p.c_str());
    if (b < 0)
    {
      RCLCPP_ERROR(logger_, "TableMagnetPlugin: post body '%s' not found.", p.c_str());
      return false;
    }
    post_bodies_.push_back(b);
    post_parent_.push_back(m->body_parentid[b]);  // the post cylinders live on the parent ("..._magnets")
  }
  if (post_bodies_.empty())
  {
    RCLCPP_ERROR(logger_, "TableMagnetPlugin: 'post_bodies' is empty.");
    return false;
  }
  for (int b = 1; b < m->nbody; ++b)
  {
    const char* n = mj_id2name(m, mjOBJ_BODY, b);
    if (!n || std::string(n).rfind(prefix, 0) != 0 || m->body_jntnum[b] != 1 ||
        m->jnt_type[m->body_jntadr[b]] != mjJNT_FREE)
    {
      continue;
    }
    const std::string eq_name = "table/" + std::string(n);
    const int eq = mj_name2id(m, mjOBJ_EQUALITY, eq_name.c_str());
    if (eq < 0 || m->eq_type[eq] != mjEQ_WELD)
    {
      RCLCPP_ERROR(logger_, "TableMagnetPlugin: weld equality '%s' missing (build_rc_cell_mjcf.py adds it).",
                   eq_name.c_str());
      return false;
    }
    plate_bodies_.push_back(b);
    table_eq_.push_back(eq);
    gripper_eq_.push_back(mj_name2id(m, mjOBJ_EQUALITY, (grip_prefix + n).c_str()));
  }

  using Trigger = std_srvs::srv::Trigger;
  auto bind = [this](Cmd c) {
    return [this, c](const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res) {
      std::tie(res->success, res->message) = call(c);
    };
  };
  on_srv_ = node_->create_service<Trigger>(on_name, bind(Cmd::On));
  off_srv_ = node_->create_service<Trigger>(off_name, bind(Cmd::Off));
  if (!clear_name.empty())
  {
    clear_srv_ = node_->create_service<Trigger>(clear_name, bind(Cmd::Clear));
  }
  state_pub_ = node_->create_publisher<std_msgs::msg::String>(state_name, rclcpp::QoS(1).transient_local());
  scratch_ = mj_makeData(m);
  RCLCPP_INFO(logger_, "TableMagnetPlugin: %zu posts, %zu plates ('%s*'), attach %.0f mm, settle %s. on=%s off=%s%s%s",
              post_bodies_.size(), plate_bodies_.size(), prefix.c_str(), attach_distance_ * 1e3, settle_ ? "on" : "off",
              on_srv_->get_service_name(), off_srv_->get_service_name(), clear_srv_ ? " clear=" : "",
              clear_srv_ ? clear_srv_->get_service_name() : "");
  return true;
}

std::pair<bool, std::string> TableMagnetPlugin::call(Cmd cmd)
{
  std::future<std::pair<bool, std::string>> fut;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_)
    {
      return { false, "previous table magnet request still pending" };
    }
    pending_.emplace(Request{ cmd, {} });
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

void TableMagnetPlugin::write_weld(const mjModel* m, int eq, const mjtNum* pos, const mjtNum* quat)
{
  // Weld data: anchor(3) on body2, relpos(3) = body2 position in body1, relquat(4) = body2 orientation in body1,
  // torquescale. body1 is the world (build_rc_cell_mjcf.py), but use its pose anyway. Caller holds the sim mutex.
  auto* mm = const_cast<mjModel*>(m);
  mjtNum* ed = mm->eq_data + mjNEQDATA * eq;
  const int b1 = m->eq_obj1id[eq];
  mjtNum dp[3], q1i[4];
  mju_sub3(dp, pos, phys_data_->xpos + 3 * b1);
  mju_zero3(ed);
  mju_mulMatTVec3(ed + 3, phys_data_->xmat + 9 * b1, dp);
  mju_negQuat(q1i, phys_data_->xquat + 4 * b1);
  mju_mulQuat(ed + 6, q1i, quat);
  ed[10] = 1.0;
}

std::string TableMagnetPlugin::held_names(const mjModel* m) const
{
  std::string s;
  for (const auto& h : held_)
  {
    s += (s.empty() ? "" : ",") + std::string(mj_id2name(m, mjOBJ_BODY, plate_bodies_[h.idx]));
  }
  return s;
}

std::pair<bool, std::string> TableMagnetPlugin::hold(const mjModel* m, mjData* d)
{
  mju_copy(scratch_->qpos, d->qpos, static_cast<int>(m->nq));
  mj_kinematics(m, scratch_);
  std::vector<mjtNum> top_z(plate_bodies_.size(), -std::numeric_limits<mjtNum>::infinity());
  std::vector<mjtNum> gap(plate_bodies_.size(), std::numeric_limits<mjtNum>::infinity());
  for (std::size_t k = 0; k < post_bodies_.size(); ++k)
  {
    const mjtNum* P = scratch_->xpos + 3 * post_bodies_[k];
    const mjtNum* R = scratch_->xmat + 9 * post_bodies_[k];
    const mjtNum X[3] = { R[0], R[3], R[6] }, Y[3] = { R[1], R[4], R[7] }, Z[3] = { R[2], R[5], R[8] };
    const mjtNum r = 0.7 * post_radius_;
    const mjtNum off[5][2] = { { 0, 0 }, { r, 0 }, { -r, 0 }, { 0, r }, { 0, -r } };
    for (const auto& o : off)
    {
      mjtNum origin[3];
      for (int i = 0; i < 3; ++i)
      {
        origin[i] = P[i] + 0.001 * Z[i] + o[0] * X[i] + o[1] * Y[i];
      }
      int hit = -1;
      const mjtNum dist = mj_ray(m, scratch_, origin, Z, nullptr, 1, post_parent_[k], &hit, nullptr);
      if (hit < 0 || dist < 0 || dist > attach_distance_)
      {
        continue;
      }
      const auto it = std::find(plate_bodies_.begin(), plate_bodies_.end(), m->geom_bodyid[hit]);
      if (it == plate_bodies_.end())
      {
        continue;
      }
      const auto i = static_cast<std::size_t>(it - plate_bodies_.begin());
      top_z[i] = std::max(top_z[i], P[2]);
      gap[i] = std::min(gap[i], dist + 0.001);
    }
  }
  std::string added;
  for (std::size_t i = 0; i < plate_bodies_.size(); ++i)
  {
    if (!std::isfinite(top_z[i]))
    {
      continue;
    }
    const bool mine = std::any_of(held_.begin(), held_.end(), [i](const Held& h) { return h.idx == static_cast<int>(i); });
    if (mine || phys_data_->eq_active[table_eq_[i]])  // already held (here or by the other table)
    {
      continue;
    }
    const int b = plate_bodies_[i];
    Held h{ static_cast<int>(i), false, {} };
    // settled pose: level, bottom face settle_clearance above the highest post under the plate
    h.target[0] = d->xpos[3 * b];
    h.target[1] = d->xpos[3 * b + 1];
    h.target[2] = top_z[i] + settle_clearance_;
    level_quat(d->xmat + 9 * b, d->xquat + 4 * b, h.target + 3);
    const bool on_gripper = gripper_eq_[i] >= 0 && d->eq_active[gripper_eq_[i]];
    if (settle_ && !on_gripper)
    {
      write_weld(m, table_eq_[i], h.target, h.target + 3);
    }
    else
    {
      write_weld(m, table_eq_[i], d->xpos + 3 * b, d->xquat + 4 * b);  // where it is now (gripper weld agrees)
      h.settle_pending = settle_;
    }
    phys_data_->eq_active[table_eq_[i]] = 1;
    held_.push_back(h);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s (gap %.1f mm, tilt %.2f deg%s)", mj_id2name(m, mjOBJ_BODY, b), gap[i] * 1e3,
                  tilt_deg(d->xmat + 9 * b), on_gripper ? ", on gripper" : "");
    added += (added.empty() ? "" : "; ") + std::string(buf);
  }
  if (held_.empty())
  {
    return { false, "no plate within " + std::to_string(attach_distance_ * 1e3) + " mm above the magnet posts" };
  }
  RCLCPP_INFO(logger_, "table magnet ON: holding %s%s%s", held_names(m).c_str(), added.empty() ? "" : " — new: ",
              added.c_str());
  return { true, "holding " + held_names(m) };
}

std::pair<bool, std::string> TableMagnetPlugin::release_all(const mjModel* m, mjData* /*d*/)
{
  const std::string names = held_names(m);
  for (const auto& h : held_)
  {
    phys_data_->eq_active[table_eq_[h.idx]] = 0;
  }
  held_.clear();
  RCLCPP_INFO(logger_, "table magnet OFF: released %s", names.empty() ? "(nothing)" : names.c_str());
  return { true, names.empty() ? "nothing held" : "released " + names };
}

std::pair<bool, std::string> TableMagnetPlugin::clear(const mjModel* m, mjData* d)
{
  if (held_.empty())
  {
    hold(m, d);  // plates resting on the posts without the magnet on
  }
  if (held_.empty())
  {
    return { false, "nothing on the table" };
  }
  auto* mm = const_cast<mjModel*>(m);
  std::string names;
  for (const auto& h : held_)
  {
    const int b = plate_bodies_[h.idx];
    phys_data_->eq_active[table_eq_[h.idx]] = 0;
    const int k = g_parked++;
    const int qa = m->jnt_qposadr[m->body_jntadr[b]];
    const int va = m->jnt_dofadr[m->body_jntadr[b]];
    mjtNum q[4];
    level_quat(d->xmat + 9 * b, d->xquat + 4 * b, q);
    for (int i = 0; i < 3; ++i)
    {
      d->qpos[qa + i] = park_position_[i] + k * park_step_[i];
    }
    mju_copy4(d->qpos + qa + 3, q);
    mju_zero(d->qvel + va, 6);
    mm->body_gravcomp[b] = 1.0;  // frozen like the inbound stack: weightless, no contacts
    for (int g = 0; g < m->ngeom; ++g)
    {
      if (m->geom_bodyid[g] == b && m->geom_group[g] != 1)
      {
        mm->geom_contype[g] = 32;
        mm->geom_conaffinity[g] = 0;
      }
    }
    mm->body_contype[b] = 32;
    mm->body_conaffinity[b] = 0;
    names += (names.empty() ? "" : ",") + std::string(mj_id2name(m, mjOBJ_BODY, b));
  }
  held_.clear();
  RCLCPP_INFO(logger_, "table clear (sim unloading stand-in): parked %s", names.c_str());
  return { true, "parked " + names };
}

void TableMagnetPlugin::update(const mjModel* m, mjData* d)
{
  {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (lock.owns_lock() && pending_)
    {
      // Same rule as MagnetGripperPlugin: eq_active / eq_data / contact bits change only between physics steps.
      std::unique_lock<std::recursive_mutex> sim_lock;
      if (sim_mutex_)
      {
        sim_lock = std::unique_lock<std::recursive_mutex>(*sim_mutex_);
      }
      mjData* pd = sim_mutex_ ? phys_data_ : d;
      std::pair<bool, std::string> r;
      switch (pending_->cmd)
      {
        case Cmd::On:
          r = hold(m, pd);
          break;
        case Cmd::Off:
          r = release_all(m, pd);
          break;
        case Cmd::Clear:
          r = clear(m, pd);
          break;
      }
      pending_->result.set_value(r);
      pending_.reset();
    }
  }

  // The 220 let go (gripper weld off) -> the magnet pulls the plate flat onto the posts.
  for (auto& h : held_)
  {
    if (!h.settle_pending || (gripper_eq_[h.idx] >= 0 && phys_data_->eq_active[gripper_eq_[h.idx]]))
    {
      continue;
    }
    std::unique_lock<std::recursive_mutex> sim_lock;
    if (sim_mutex_)
    {
      sim_lock = std::unique_lock<std::recursive_mutex>(*sim_mutex_);
    }
    const int b = plate_bodies_[h.idx];
    const double dz = (phys_data_->xpos[3 * b + 2] - h.target[2]) * 1e3;
    const double tilt = tilt_deg(phys_data_->xmat + 9 * b);
    write_weld(m, table_eq_[h.idx], h.target, h.target + 3);
    h.settle_pending = false;
    RCLCPP_INFO(logger_, "table magnet: %s released by the gripper -> pulled flat onto the posts (down %.1f mm, tilt "
                         "%.2f -> 0 deg)",
                mj_id2name(m, mjOBJ_BODY, b), dz, tilt);
  }

  if (d->time - last_state_pub_ >= 0.5 || d->time < last_state_pub_)
  {
    last_state_pub_ = d->time;
    std_msgs::msg::String msg;
    msg.data = held_names(m);
    state_pub_->publish(msg);
  }
}

void TableMagnetPlugin::cleanup()
{
  if (scratch_)
  {
    mj_deleteData(scratch_);
    scratch_ = nullptr;
  }
  on_srv_.reset();
  off_srv_.reset();
  clear_srv_.reset();
  state_pub_.reset();
  node_.reset();
}

}  // namespace mujoco_ros2_control_plugins

PLUGINLIB_EXPORT_CLASS(mujoco_ros2_control_plugins::TableMagnetPlugin,
                       mujoco_ros2_control_plugins::MuJoCoROS2ControlPluginBase)
