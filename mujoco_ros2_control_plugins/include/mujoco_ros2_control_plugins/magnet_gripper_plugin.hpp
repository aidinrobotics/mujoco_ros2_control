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

#ifndef MUJOCO_ROS2_CONTROL_PLUGINS__MAGNET_GRIPPER_PLUGIN_HPP_
#define MUJOCO_ROS2_CONTROL_PLUGINS__MAGNET_GRIPPER_PLUGIN_HPP_

#include <future>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "mujoco_ros2_control_plugins/mujoco_ros2_control_plugins_base.hpp"

namespace mujoco_ros2_control_plugins
{

/**
 * @brief Magnet gripper: on command, the plate under the magnet face sticks to the gripper; off releases it.
 *
 * magnet_on  (std_srvs/Trigger): among bodies whose name starts with plate_prefix, find the plate whose geoms are
 *            within attach_distance of any magnet cell geom of gripper_body (mj_geomDistance on a private mjData).
 *            The nearest one is attached by activating the MJCF weld "magnet/<plate>" (gripper -> plate, inactive
 *            in the model) with the current relative pose written into its eq_data. While carried the plate is
 *            weightless (gravcomp 1) and has no contacts. (A spring-damper wrench was tried first: the physics and
 *            control loops run separately, the wrench was computed from stale/torn data and the plate diverged.)
 * magnet_off (std_srvs/Trigger): the carried plate is released with gravity and full contacts (released_piece /
 *            released_sphere bits), so it settles by physics.
 *            If the plate is released inside frozen_zone [xmin, xmax, ymin, ymax] (the inbound pallet — a plate put back
 *            on the stack), it is frozen there like the rest of the static stack (gravcomp 1, contact bits frozen_bits)
 *            instead: the static stack plates have no contacts, so a physical plate would fall through them.
 * state_topic (std_msgs/String): name of the carried plate, "" when empty.
 * Plate poses (ground truth) go to /tf at tf_rate: world -> <tf_prefix><plate name without plate_prefix>, the body
 * frame (bottom face at the CoG, +Z up through the plate). A vision node can later publish the same frames.
 *
 * Gravity / contact bits and the weld's relpose are model fields (shared with the physics thread); eq_active is in
 * the physics mjData given to init().
 *
 * Per-cell (channel) magnets (2026-10-19): the real gripper switches each of its 2x2 cells on/off (vision PickPlan
 * magnet_channel ch0..ch3). cell_geoms are given in channel order. cells_service (rcl_interfaces/SetParameters, one
 * parameter "channels" = integer_array [ch0, ch1, ch2, ch3] of 0/1, or a string "1,1,0,1") selects the enabled cells;
 * magnet_on counts only enabled cells. An enabled cell whose ray hits ANOTHER plate within warn_distance (at magnet_on
 * and every check_period while holding) is logged as a warning (cell, plate, distance mm) — a cell left on over a
 * neighbouring plate would pull that plate. If a channel change leaves no enabled cell on the carried plate it drops.
 * cells_state_topic (std_msgs/String, 10 Hz): "channels=1,1,1,1 held=<plate> on_plate=ch0,ch1 spring_mm=.. max_mm=..
 * warn=ch2>plate@12.3mm" (magnet_state stays the plain plate name — other nodes parse it).
 *
 * Carried plate (2026-10-19, gripper spring): carried_piece / carried_sphere contact bits and carried_gravcomp while
 * held (default 0/0, 0/0, 1 = weightless and contact-free as before). With the spring the support spheres touch the
 * equipment ([8, 0]) so the plate pressed on the grinding table posts pushes the spring in, and the plate keeps its
 * weight (0). spring_joint (optional): the face slide joint — its displacement is reported in the cells state.
 *
 * Parameters (mujoco_plugins.<name>.):
 *   gripper_body, cell_geoms (names of the magnet cell collision geoms, channel order), plate_prefix,
 *   attach_distance [m], released_piece [contype, conaffinity], released_sphere [contype, conaffinity], frozen_zone
 *   [xmin,xmax,ymin,ymax] (empty = off), frozen_bits [contype, conaffinity], carried_piece, carried_sphere,
 *   carried_gravcomp, min_cells (enabled cells on the plate needed to attach, default 1), warn_distance [m],
 *   check_period [s], spring_joint, on_service, off_service, state_topic, cells_service, cells_state_topic
 *   (absolute names recommended)
 */
class MagnetGripperPlugin : public MuJoCoROS2ControlPluginBase
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
    Cells
  };
  struct Request
  {
    Cmd cmd;
    std::vector<int> channels;
    std::promise<std::pair<bool, std::string>> result;
  };
  struct CellHit
  {
    int body{ -1 };       // body of the first geom the cell's ray hits (-1 = nothing)
    double gap{ -1.0 };   // face -> hit distance [m]
  };

  std::pair<bool, std::string> attach(const mjModel* m, mjData* d);
  std::pair<bool, std::string> release(const mjModel* m);
  std::pair<bool, std::string> set_channels(const mjModel* m, const std::vector<int>& ch);
  void set_plate_contacts(const mjModel* m, int body, int piece_ct, int piece_ca, int sph_ct, int sph_ca);
  std::pair<bool, std::string> call(Cmd cmd, std::vector<int> channels = {});
  std::vector<CellHit> cast_cells(const mjModel* m, const mjData* d);   // rays from all cells (kinematics on scratch_)
  bool is_plate(int b) const;
  std::string cell_label(std::size_t k) const;
  // enabled cells over another plate (not `own`) within warn_distance -> warning text, logs new ones
  std::string check_neighbours(const mjModel* m, const std::vector<CellHit>& hits, int own, bool log_all);
  std::string channels_text() const;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("MagnetGripperPlugin") };

  int gripper_body_{ -1 };
  std::vector<int> cell_geoms_;
  std::vector<int> plate_bodies_;
  double attach_distance_{ 0.01 };
  std::vector<int64_t> released_piece_{ 1, 15 };
  std::vector<int64_t> released_sphere_{ 8, 0 };
  std::vector<double> frozen_zone_;
  std::vector<int64_t> frozen_bits_{ 32, 0 };
  std::vector<int64_t> carried_piece_{ 0, 0 };
  std::vector<int64_t> carried_sphere_{ 0, 0 };
  double carried_gravcomp_{ 1.0 };
  int min_cells_{ 1 };
  double warn_distance_{ 0.03 };
  double check_period_{ 0.2 };
  std::vector<int> enabled_;            // per cell_geoms_ entry (channel), 0/1
  std::vector<std::string> cell_names_; // "cell4" … (from the geom name suffix)
  int spring_qadr_{ -1 };
  double spring_max_{ 0.0 };            // [m] since the last magnet_on / magnet_off
  std::set<std::pair<int, int>> warned_;   // (channel, plate) already logged during this hold
  std::string warn_text_, on_plate_text_;
  double last_check_{ -1.0 };

  // carried plate (update() thread only)
  int carried_{ -1 };
  int carried_eq_{ -1 };
  std::vector<int> plate_eq_;     // weld "magnet/<plate>" per plate_bodies_ entry
  mjData* phys_data_{ nullptr };

  std::mutex mutex_;
  std::optional<Request> pending_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr on_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr off_srv_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Service<rcl_interfaces::srv::SetParameters>::SharedPtr cells_srv_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cells_state_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  tf2_msgs::msg::TFMessage tf_msg_;
  double tf_period_{ 0.1 };
  double last_tf_pub_{ -1.0 };
  double last_state_pub_{ -1.0 };
  mjData* scratch_{ nullptr };   // private data for mj_geomDistance (control data is shared with the physics copy)
};

}  // namespace mujoco_ros2_control_plugins

#endif  // MUJOCO_ROS2_CONTROL_PLUGINS__MAGNET_GRIPPER_PLUGIN_HPP_
