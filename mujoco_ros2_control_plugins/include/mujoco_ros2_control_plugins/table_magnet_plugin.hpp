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

#ifndef MUJOCO_ROS2_CONTROL_PLUGINS__TABLE_MAGNET_PLUGIN_HPP_
#define MUJOCO_ROS2_CONTROL_PLUGINS__TABLE_MAGNET_PLUGIN_HPP_

#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace mujoco_ros2_control_plugins
{

/**
 * @brief Table magnets (grinding table magnet posts): on command, every plate lying on the posts is held in place;
 *        off lets it go. Real table magnets are very strong — the 220 hands a plate over only after they hold it.
 *
 * magnet_on  (std_srvs/Trigger): rays go up from each post top (centre + 4 points at 0.7 r). A plate body (name starts
 *            with plate_prefix) hit within attach_distance is held: the MJCF weld "table/<plate>" (world -> plate,
 *            inactive in the model, build_rc_cell_mjcf.py adds it) gets the plate's current world pose and is switched
 *            on — so a plate still hanging on the 220 gripper weld is not pulled (both welds agree). success = at least
 *            one plate held (message = names); nothing on the posts -> success false.
 *            With settle (default true) the magnet then pulls the plate flat onto the posts as soon as the gripper weld
 *            "<gripper_weld_prefix><plate>" is off: the weld target becomes the plate pose with roll/pitch removed and
 *            its bottom face settle_clearance above the highest post under it (models "magnet pulls the plate down").
 * magnet_off (std_srvs/Trigger): all welds of this table off; the plates are ordinary physics again.
 * clear_service (std_srvs/Trigger, optional, sim stand-in for an unloading robot): held plates are released and moved
 *            to park_position + k * park_step (k counts parked plates of all tables), frozen there (gravcomp 1, no
 *            contacts) — used until the unloading draft exists.
 * state_topic (std_msgs/String): held plate names, comma separated ("" = none).
 *
 * Parameters (mujoco_plugins.<name>.):
 *   post_bodies (frames at the post tops, +Z up), post_radius [m], plate_prefix, attach_distance [m], settle,
 *   settle_clearance [m], gripper_weld_prefix, on_service, off_service, state_topic, clear_service,
 *   park_position [x,y,z], park_step [x,y,z]
 */
class TableMagnetPlugin : public MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override;
  void update(const mjModel* model, mjData* data) override;
  void cleanup() override;

private:
  enum class Cmd
  {
    On,
    Off,
    Clear
  };
  struct Request
  {
    Cmd cmd;
    std::promise<std::pair<bool, std::string>> result;
  };
  struct Held
  {
    int idx;              // index into plate_bodies_
    bool settle_pending;  // waiting for the gripper weld to go off before pulling the plate flat
    mjtNum target[7];     // settled pose (pos, quat) in world
  };

  std::pair<bool, std::string> hold(const mjModel* m, mjData* d);
  std::pair<bool, std::string> release_all(const mjModel* m, mjData* d);
  std::pair<bool, std::string> clear(const mjModel* m, mjData* d);
  void write_weld(const mjModel* m, int eq, const mjtNum* pos, const mjtNum* quat);
  std::pair<bool, std::string> call(Cmd cmd);
  std::string held_names(const mjModel* m) const;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("TableMagnetPlugin") };

  std::vector<int> post_bodies_;
  std::vector<int> post_parent_;
  double post_radius_{ 0.05 };
  double attach_distance_{ 0.03 };
  bool settle_{ true };
  double settle_clearance_{ 0.0005 };
  std::vector<int> plate_bodies_;
  std::vector<int> table_eq_;    // weld "table/<plate>" per plate
  std::vector<int> gripper_eq_;  // weld "<gripper_weld_prefix><plate>" per plate (-1 = none)
  std::vector<double> park_position_{ 0.83, -1.75, 0.25 };
  std::vector<double> park_step_{ 0.0, 0.0, 0.03 };

  std::vector<Held> held_;  // update() thread only
  mjData* phys_data_{ nullptr };
  mjData* scratch_{ nullptr };

  std::mutex mutex_;
  std::optional<Request> pending_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr on_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr off_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  double last_state_pub_{ -1.0 };
};

}  // namespace mujoco_ros2_control_plugins

#endif  // MUJOCO_ROS2_CONTROL_PLUGINS__TABLE_MAGNET_PLUGIN_HPP_
