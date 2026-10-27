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
#include <cstdio>
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
  frozen_weld_prefix_ = declare_or_get<std::string>(node_, "frozen_weld_prefix", "");
  carried_piece_ = declare_or_get<std::vector<int64_t>>(node_, "carried_piece", { 0, 0 });
  carried_sphere_ = declare_or_get<std::vector<int64_t>>(node_, "carried_sphere", { 0, 0 });
  carried_gravcomp_ = declare_or_get<double>(node_, "carried_gravcomp", 1.0);
  min_cells_ = static_cast<int>(declare_or_get<int64_t>(node_, "min_cells", 1));
  warn_distance_ = declare_or_get<double>(node_, "warn_distance", 0.03);
  check_period_ = declare_or_get<double>(node_, "check_period", 0.2);
  const auto spring_joint = declare_or_get<std::string>(node_, "spring_joint", "");
  const auto cells_name = declare_or_get<std::string>(node_, "cells_service", "magnet_cells");
  const auto cells_state_name = declare_or_get<std::string>(node_, "cells_state_topic", "magnet_cells_state");
  if (carried_piece_.size() != 2 || carried_sphere_.size() != 2)
  {
    RCLCPP_ERROR(logger_, "MagnetGripperPlugin: carried_piece / carried_sphere need [contype, conaffinity].");
    return false;
  }
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
    const auto us = c.rfind('_');
    cell_names_.push_back("cell" + (us == std::string::npos ? std::to_string(cell_geoms_.size() - 1) : c.substr(us + 1)));
  }
  enabled_.assign(cell_geoms_.size(), 1);
  if (!spring_joint.empty())
  {
    const int j = mj_name2id(m, mjOBJ_JOINT, spring_joint.c_str());
    if (j < 0)
    {
      RCLCPP_ERROR(logger_, "MagnetGripperPlugin: spring_joint '%s' not found.", spring_joint.c_str());
      return false;
    }
    spring_qadr_ = m->jnt_qposadr[j];
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
        std::tie(res->success, res->message) = call(Cmd::On);
      });
  off_srv_ = node_->create_service<std_srvs::srv::Trigger>(
      off_name, [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        std::tie(res->success, res->message) = call(Cmd::Off);
      });
  // Cell selection: SetParameters with one parameter "channels" (integer_array / bool_array of 0/1 in channel order,
  // or a string "1,1,0,1"). A standard type so that nothing here depends on the cell's own message package.
  using SetParameters = rcl_interfaces::srv::SetParameters;
  cells_srv_ = node_->create_service<SetParameters>(
      cells_name, [this](const std::shared_ptr<SetParameters::Request> req, std::shared_ptr<SetParameters::Response> res) {
        rcl_interfaces::msg::SetParametersResult r;
        std::vector<int> ch;
        bool found = false;
        for (const auto& p : req->parameters)
        {
          if (p.name != "channels")
          {
            continue;
          }
          found = true;
          const auto& v = p.value;
          if (v.type == rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER_ARRAY)
          {
            for (auto x : v.integer_array_value)
              ch.push_back(x != 0);
          }
          else if (v.type == rcl_interfaces::msg::ParameterType::PARAMETER_BOOL_ARRAY)
          {
            for (auto x : v.bool_array_value)
              ch.push_back(x ? 1 : 0);
          }
          else if (v.type == rcl_interfaces::msg::ParameterType::PARAMETER_STRING)
          {
            for (char c : v.string_value)
            {
              if (c == '0' || c == '1')
                ch.push_back(c - '0');
            }
          }
        }
        if (!found || ch.size() != cell_geoms_.size())
        {
          r.successful = false;
          r.reason = "need parameter 'channels' with " + std::to_string(cell_geoms_.size()) +
                     " values 0/1 (channel order ch0..)";
        }
        else
        {
          std::tie(r.successful, r.reason) = call(Cmd::Cells, ch);
        }
        res->results.assign(std::max<std::size_t>(1, req->parameters.size()), r);
      });
  state_pub_ = node_->create_publisher<std_msgs::msg::String>(state_name, rclcpp::QoS(1).transient_local());
  cells_state_pub_ = node_->create_publisher<std_msgs::msg::String>(cells_state_name, rclcpp::QoS(1).transient_local());
  scratch_ = mj_makeData(m);
  tf_pub_ = node_->create_publisher<tf2_msgs::msg::TFMessage>("/tf", 10);
  for (int b : plate_bodies_)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.frame_id = "world";
    t.child_frame_id = tf_prefix + std::string(mj_id2name(m, mjOBJ_BODY, b)).substr(prefix.size());
    tf_msg_.transforms.push_back(t);
  }

  std::string order;
  for (std::size_t k = 0; k < cell_geoms_.size(); ++k)
  {
    order += (k ? " " : "") + cell_label(k);
  }
  RCLCPP_INFO(logger_,
              "MagnetGripperPlugin: gripper '%s', %zu cells [%s], %zu plates ('%s*'). on=%s off=%s state=%s cells=%s "
              "cells_state=%s, carried bits piece %ld/%ld sphere %ld/%ld gravcomp %.1f, spring %s",
              gripper.c_str(), cell_geoms_.size(), order.c_str(), plate_bodies_.size(), prefix.c_str(),
              on_srv_->get_service_name(), off_srv_->get_service_name(), state_pub_->get_topic_name(),
              cells_srv_->get_service_name(), cells_state_pub_->get_topic_name(), carried_piece_[0], carried_piece_[1],
              carried_sphere_[0], carried_sphere_[1], carried_gravcomp_, spring_qadr_ >= 0 ? spring_joint.c_str() : "-");
  return true;
}

std::pair<bool, std::string> MagnetGripperPlugin::call(Cmd cmd, std::vector<int> channels)
{
  std::future<std::pair<bool, std::string>> fut;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_)
    {
      return { false, "previous magnet request still pending" };
    }
    pending_.emplace(Request{ cmd, std::move(channels), {} });
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
  // Each cell shoots a ray out of its face along the tool axis (gripper body +X); the first thing it hits must be a
  // plate within attach_distance. Closest-geometry distance also caught a neighbouring plate's raised edge next to
  // the face (tilted ga1_03 beside ga2_04) and welded that plate at a large offset. Only ENABLED cells count; the plate
  // under the most of them wins, ties broken by the smallest gap.
  const auto hits = cast_cells(m, d);
  std::vector<int> n_cells(m->nbody, 0);
  std::vector<mjtNum> gap(m->nbody, std::numeric_limits<mjtNum>::infinity());
  int n_enabled = 0;
  for (std::size_t k = 0; k < hits.size(); ++k)
  {
    n_enabled += enabled_[k];
    const auto& h = hits[k];
    if (!enabled_[k] || h.body < 0 || h.gap > attach_distance_ || !is_plate(h.body))
    {
      continue;
    }
    ++n_cells[h.body];
    gap[h.body] = std::min<mjtNum>(gap[h.body], h.gap);
  }
  if (n_enabled == 0)
  {
    return { false, "no magnet cell enabled (" + channels_text() + ")" };
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
    return { false, "no plate within " + std::to_string(attach_distance_ * 1e3) +
                        " mm in front of the enabled magnet cells (" + channels_text() + ")" };
  }
  if (n_cells[best] < min_cells_)
  {
    return { false, std::string("only ") + std::to_string(n_cells[best]) + " enabled cell(s) on " +
                        mj_id2name(m, mjOBJ_BODY, best) + " (min_cells " + std::to_string(min_cells_) + ")" };
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

  // While carried: carried_gravcomp (1 = weightless) and carried_* contact bits (default none).
  mm->body_gravcomp[best] = carried_gravcomp_;
  set_plate_contacts(m, best, static_cast<int>(carried_piece_[0]), static_cast<int>(carried_piece_[1]),
                     static_cast<int>(carried_sphere_[0]), static_cast<int>(carried_sphere_[1]));
  carried_eq_ = eq;
  phys_data_->eq_active[eq] = 1;
  carried_ = best;
  for (auto it = frozen_welds_.begin(); it != frozen_welds_.end(); ++it)   // picked up again: drop its world weld
  {
    if (m->eq_obj2id[*it] == best)
    {
      phys_data_->eq_active[*it] = 0;
      frozen_welds_.erase(it);
      break;
    }
  }
  const std::string name = mj_id2name(m, mjOBJ_BODY, best);
  // which enabled cells hold it, which are off the plate, which are over another plate
  on_plate_text_.clear();
  std::string off_plate;
  for (std::size_t k = 0; k < hits.size(); ++k)
  {
    if (!enabled_[k])
      continue;
    if (hits[k].body == best && hits[k].gap <= attach_distance_)
      on_plate_text_ += (on_plate_text_.empty() ? "" : ",") + cell_label(k);
    else
      off_plate += (off_plate.empty() ? "" : ",") + cell_label(k);
  }
  warned_.clear();
  spring_max_ = 0.0;
  warn_text_ = check_neighbours(m, hits, best, true);
  RCLCPP_INFO(logger_, "magnet ON: %s (%d/%d enabled cells: %s, gap %.1f mm; %s)", name.c_str(), best_cells, n_enabled,
              on_plate_text_.c_str(), std::max<mjtNum>(best_dist, 0) * 1e3, channels_text().c_str());
  if (!off_plate.empty())
  {
    RCLCPP_WARN(logger_, "magnet ON: enabled cell(s) %s not on %s (off the plate — turn them off)", off_plate.c_str(),
                name.c_str());
  }
  if (best_cells < 2)
  {
    RCLCPP_WARN(logger_, "magnet ON: %s held by %d cell only (real gripper: 200 kg/cell, at least 2 cells)",
                name.c_str(), best_cells);
  }
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
    const int weq = frozen_weld_prefix_.empty() ? -1 :
                    mj_name2id(m, mjOBJ_EQUALITY, (frozen_weld_prefix_ + name).c_str());
    if (weq >= 0 && m->eq_type[weq] == mjEQ_WELD && m->eq_obj2id[weq] == carried_)
    {
      // weld to body1 (the world) at the pose it has now
      auto* mm = const_cast<mjModel*>(m);
      mjtNum* ed = mm->eq_data + mjNEQDATA * weq;
      const int b1 = m->eq_obj1id[weq];
      mjtNum dp[3], q1i[4];
      mju_sub3(dp, phys_data_->xpos + 3 * carried_, phys_data_->xpos + 3 * b1);
      mju_zero3(ed);
      mju_mulMatTVec3(ed + 3, phys_data_->xmat + 9 * b1, dp);
      mju_negQuat(q1i, phys_data_->xquat + 4 * b1);
      mju_mulQuat(ed + 6, q1i, phys_data_->xquat + 4 * carried_);
      ed[10] = 1.0;
      phys_data_->eq_active[weq] = 1;
      frozen_welds_.push_back(weq);
    }
  }
  else
  {
    const_cast<mjModel*>(m)->body_gravcomp[carried_] = 0.0;
    set_plate_contacts(m, carried_, static_cast<int>(released_piece_.at(0)), static_cast<int>(released_piece_.at(1)),
                       static_cast<int>(released_sphere_.at(0)), static_cast<int>(released_sphere_.at(1)));
  }
  carried_ = -1;
  on_plate_text_.clear();
  warn_text_.clear();
  warned_.clear();
  RCLCPP_INFO(logger_, "magnet OFF: released %s%s (spring max %.1f mm while held)", name.c_str(),
              frozen ? " (frozen zone: kept static on the stack)" : "", spring_max_ * 1e3);
  spring_max_ = 0.0;
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
      std::pair<bool, std::string> r;
      switch (pending_->cmd)
      {
        case Cmd::On:
          r = attach(m, pd);
          break;
        case Cmd::Off:
          r = release(m);
          break;
        case Cmd::Cells:
          r = set_channels(m, pending_->channels);
          break;
      }
      pending_->result.set_value(r);
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

  if (spring_qadr_ >= 0)
  {
    spring_max_ = std::max<double>(spring_max_, d->qpos[spring_qadr_]);
  }
  // while holding: enabled cells over another plate (a neighbouring / lower plate the magnet would pull)
  if (carried_ >= 0 && check_period_ > 0 && (d->time - last_check_ >= check_period_ || d->time < last_check_))
  {
    last_check_ = d->time;
    warn_text_ = check_neighbours(m, cast_cells(m, d), carried_, false);
  }

  if (d->time - last_state_pub_ >= 0.1 || d->time < last_state_pub_)
  {
    last_state_pub_ = d->time;
    std_msgs::msg::String msg;
    msg.data = carried_ >= 0 ? mj_id2name(m, mjOBJ_BODY, carried_) : "";
    state_pub_->publish(msg);
    char buf[96];
    std::snprintf(buf, sizeof(buf), " spring_mm=%.2f max_mm=%.2f",
                  spring_qadr_ >= 0 ? d->qpos[spring_qadr_] * 1e3 : 0.0, spring_max_ * 1e3);
    std_msgs::msg::String cs;
    cs.data = "channels=" + channels_text() + " held=" + msg.data + " on_plate=" + on_plate_text_ + buf +
              " warn=" + warn_text_;
    cells_state_pub_->publish(cs);
  }
}

std::vector<MagnetGripperPlugin::CellHit> MagnetGripperPlugin::cast_cells(const mjModel* m, const mjData* d)
{
  // Measure on a private mjData: mj_ray / mj_kinematics must not touch the physics data's state or arena.
  mju_copy(scratch_->qpos, d->qpos, static_cast<int>(m->nq));
  mj_kinematics(m, scratch_);
  const mjtNum* gx = scratch_->xmat + 9 * gripper_body_;
  const mjtNum dir[3] = { gx[0], gx[3], gx[6] };
  std::vector<CellHit> out(cell_geoms_.size());
  for (std::size_t k = 0; k < cell_geoms_.size(); ++k)
  {
    const int c = cell_geoms_[k];
    // centre -> face distance along dir (box half extents projected on dir)
    const mjtNum* cx = scratch_->geom_xmat + 9 * c;
    mjtNum half = 0;
    for (int i = 0; i < 3; ++i)
    {
      half += m->geom_size[3 * c + i] * std::abs(cx[i] * dir[0] + cx[3 + i] * dir[1] + cx[6 + i] * dir[2]);
    }
    int hit = -1;
    const mjtNum dist = mj_ray(m, scratch_, scratch_->geom_xpos + 3 * c, dir, nullptr, 1, gripper_body_, &hit, nullptr);
    if (hit >= 0 && dist >= 0)
    {
      out[k].body = m->geom_bodyid[hit];
      out[k].gap = dist - half;
    }
  }
  return out;
}

bool MagnetGripperPlugin::is_plate(int b) const
{
  return std::find(plate_bodies_.begin(), plate_bodies_.end(), b) != plate_bodies_.end();
}

std::string MagnetGripperPlugin::cell_label(std::size_t k) const
{
  return "ch" + std::to_string(k) + "(" + cell_names_[k] + ")";
}

std::string MagnetGripperPlugin::channels_text() const
{
  std::string s;
  for (std::size_t k = 0; k < enabled_.size(); ++k)
  {
    s += (k ? "," : "") + std::to_string(enabled_[k]);
  }
  return s;
}

std::string MagnetGripperPlugin::check_neighbours(const mjModel* m, const std::vector<CellHit>& hits, int own,
                                                  bool log_all)
{
  std::string text;
  for (std::size_t k = 0; k < hits.size(); ++k)
  {
    const auto& h = hits[k];
    if (!enabled_[k] || h.body < 0 || h.body == own || !is_plate(h.body) || h.gap > warn_distance_)
    {
      continue;
    }
    const std::string plate = mj_id2name(m, mjOBJ_BODY, h.body);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s>%s@%.1fmm", ("ch" + std::to_string(k)).c_str(),
                  plate.substr(plate.rfind('/') + 1).c_str(), std::max(0.0, h.gap) * 1e3);
    text += (text.empty() ? "" : ";") + std::string(buf);
    if (log_all || warned_.insert({ static_cast<int>(k), h.body }).second)
    {
      warned_.insert({ static_cast<int>(k), h.body });
      RCLCPP_WARN(logger_, "magnet: enabled cell %s is over another plate %s (%.1f mm) — it would pull that plate; "
                           "turn this cell off",
                  cell_label(k).c_str(), plate.c_str(), std::max(0.0, h.gap) * 1e3);
    }
  }
  return text;
}

std::pair<bool, std::string> MagnetGripperPlugin::set_channels(const mjModel* m, const std::vector<int>& ch)
{
  const std::string before = channels_text();
  enabled_.assign(ch.begin(), ch.end());
  std::string msg = "channels " + before + " -> " + channels_text();
  if (carried_ >= 0)
  {
    // Switching cells while holding: if no enabled cell is left on the carried plate it drops (magnet off).
    const auto hits = cast_cells(m, phys_data_);
    int on = 0;
    on_plate_text_.clear();
    for (std::size_t k = 0; k < hits.size(); ++k)
    {
      if (enabled_[k] && hits[k].body == carried_ && hits[k].gap <= attach_distance_ + 0.005)
      {
        ++on;
        on_plate_text_ += (on_plate_text_.empty() ? "" : ",") + cell_label(k);
      }
    }
    if (on == 0)
    {
      RCLCPP_ERROR(logger_, "magnet: %s leaves no enabled cell on %s — the plate drops", msg.c_str(),
                   mj_id2name(m, mjOBJ_BODY, carried_));
      msg += "; no cell left on the plate: " + release(m).second + " dropped";
    }
    else
    {
      warned_.clear();
      warn_text_ = check_neighbours(m, hits, carried_, true);
    }
  }
  RCLCPP_INFO(logger_, "magnet cells: %s", msg.c_str());
  return { true, msg };
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
