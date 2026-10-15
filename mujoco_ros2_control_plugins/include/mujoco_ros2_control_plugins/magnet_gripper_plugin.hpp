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
#include <string>
#include <vector>

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
 * state_topic (std_msgs/String): name of the carried plate, "" when empty.
 * Plate poses (ground truth) go to /tf at tf_rate: world -> <tf_prefix><plate name without plate_prefix>, the body
 * frame (bottom face at the CoG, +Z up through the plate). A vision node can later publish the same frames.
 *
 * Gravity / contact bits and the weld's relpose are model fields (shared with the physics thread); eq_active is in
 * the physics mjData given to init().
 *
 * Parameters (mujoco_plugins.<name>.):
 *   gripper_body, cell_geoms (names of the magnet cell collision geoms), plate_prefix, attach_distance [m],
 *   released_piece [contype, conaffinity], released_sphere [contype, conaffinity],
 *   on_service, off_service, state_topic (absolute names recommended)
 */
class MagnetGripperPlugin : public MuJoCoROS2ControlPluginBase
{
public:
  bool init(rclcpp::Node::SharedPtr node, const mjModel* model, mjData* data) override;
  void update(const mjModel* model, mjData* data) override;
  void cleanup() override;

private:
  struct Request
  {
    bool on;
    std::promise<std::pair<bool, std::string>> result;
  };

  std::pair<bool, std::string> attach(const mjModel* m, mjData* d);
  std::pair<bool, std::string> release(const mjModel* m);
  void set_plate_contacts(const mjModel* m, int body, int piece_ct, int piece_ca, int sph_ct, int sph_ca);
  std::pair<bool, std::string> call(bool on);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("MagnetGripperPlugin") };

  int gripper_body_{ -1 };
  std::vector<int> cell_geoms_;
  std::vector<int> plate_bodies_;
  double attach_distance_{ 0.01 };
  std::vector<int64_t> released_piece_{ 1, 15 };
  std::vector<int64_t> released_sphere_{ 8, 0 };

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
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  tf2_msgs::msg::TFMessage tf_msg_;
  double tf_period_{ 0.1 };
  double last_tf_pub_{ -1.0 };
  double last_state_pub_{ -1.0 };
  mjData* scratch_{ nullptr };   // private data for mj_geomDistance (control data is shared with the physics copy)
};

}  // namespace mujoco_ros2_control_plugins

#endif  // MUJOCO_ROS2_CONTROL_PLUGINS__MAGNET_GRIPPER_PLUGIN_HPP_
