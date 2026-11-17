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

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>

#include "mujoco_ros2_control_plugins/magnet_gripper_plugin.hpp"
#include "mujoco_ros2_control_plugins/table_magnet_plugin.hpp"

namespace
{
// One stacked plate (gravcomp, frozen contact bits) under a magnet gripper, next to a table with one post. The plate
// has the gripper weld magnet/<plate> and the world weld table/<plate>, both off at the start.
constexpr const char* kMjcf = R"(
<mujoco model="magnet_reset_test">
  <worldbody>
    <body name="gripper" pos="0 0 1">
      <geom name="cell_1" type="box" size="0.05 0.05 0.01" contype="0" conaffinity="0"/>
    </body>
    <body name="table_magnets" pos="1 0 0">
      <body name="post_1" pos="0 0 0.1">
        <geom type="cylinder" size="0.05 0.1"/>
      </body>
    </body>
    <body name="inbound/p1" pos="0 0 0.5" gravcomp="1">
      <freejoint/>
      <geom name="p1_box" type="box" size="0.2 0.2 0.01" contype="32" conaffinity="8"/>
    </body>
  </worldbody>
  <equality>
    <weld name="magnet/inbound/p1" body1="gripper" body2="inbound/p1" active="false"/>
    <weld name="table/inbound/p1" body1="world" body2="inbound/p1" active="false"/>
  </equality>
</mujoco>
)";
}  // namespace

class MagnetResetTest : public ::testing::Test
{
protected:
  static void SetUpTestCase()
  {
    if (!rclcpp::ok())
    {
      rclcpp::init(0, nullptr);
    }
  }

  static void TearDownTestCase()
  {
    if (rclcpp::ok())
    {
      rclcpp::shutdown();
    }
  }

  void SetUp() override
  {
    char error[1000] = "";
    mjSpec* spec = mj_parseXMLString(kMjcf, nullptr, error, sizeof(error));
    ASSERT_NE(spec, nullptr) << error;
    model_ = mj_compile(spec, nullptr);
    mj_deleteSpec(spec);
    ASSERT_NE(model_, nullptr);
    data_ = mj_makeData(model_);
    mj_forward(model_, data_);
    node_ = std::make_shared<rclcpp::Node>("magnet_reset_test_node");
    plate_ = mj_name2id(model_, mjOBJ_BODY, "inbound/p1");
    geom_ = mj_name2id(model_, mjOBJ_GEOM, "p1_box");
    magnet_eq_ = mj_name2id(model_, mjOBJ_EQUALITY, "magnet/inbound/p1");
    table_eq_ = mj_name2id(model_, mjOBJ_EQUALITY, "table/inbound/p1");
  }

  void TearDown() override
  {
    mj_deleteData(data_);
    mj_deleteModel(model_);
  }

  mjModel* model_{ nullptr };
  mjData* data_{ nullptr };
  rclcpp::Node::SharedPtr node_;
  int plate_{ -1 }, geom_{ -1 }, magnet_eq_{ -1 }, table_eq_{ -1 };
};

TEST_F(MagnetResetTest, GripperResetPutsThePlateContactsGravcompAndWeldsBackToTheStart)
{
  auto sub = node_->create_sub_node("gripper");
  sub->declare_parameter("mujoco_plugins.gripper.gripper_body", std::string("gripper"));
  sub->declare_parameter("mujoco_plugins.gripper.cell_geoms", std::vector<std::string>{ "cell_1" });
  sub->declare_parameter("mujoco_plugins.gripper.frozen_weld_prefix", std::string("table/"));
  mujoco_ros2_control_plugins::MagnetGripperPlugin plugin;
  ASSERT_TRUE(plugin.init(sub, model_, data_));

  // what a pick and a release on a table leave behind
  model_->geom_contype[geom_] = 1;
  model_->geom_conaffinity[geom_] = 15;
  model_->body_contype[plate_] = 1;
  model_->body_conaffinity[plate_] = 15;
  model_->body_gravcomp[plate_] = 0.0;
  data_->eq_active[magnet_eq_] = 1;
  data_->eq_active[table_eq_] = 1;

  plugin.reset(model_, data_);

  EXPECT_EQ(model_->geom_contype[geom_], 32);
  EXPECT_EQ(model_->geom_conaffinity[geom_], 8);
  EXPECT_EQ(model_->body_contype[plate_], 32);
  EXPECT_EQ(model_->body_conaffinity[plate_], 8);
  EXPECT_DOUBLE_EQ(model_->body_gravcomp[plate_], 1.0);
  EXPECT_EQ(data_->eq_active[magnet_eq_], 0);
  EXPECT_EQ(data_->eq_active[table_eq_], 0);
  plugin.cleanup();
}

TEST_F(MagnetResetTest, TableResetSwitchesTheTableWeldsBackToTheStart)
{
  auto sub = node_->create_sub_node("table");
  sub->declare_parameter("mujoco_plugins.table.post_bodies", std::vector<std::string>{ "post_1" });
  mujoco_ros2_control_plugins::TableMagnetPlugin plugin;
  ASSERT_TRUE(plugin.init(sub, model_, data_));

  data_->eq_active[table_eq_] = 1;

  plugin.reset(model_, data_);

  EXPECT_EQ(data_->eq_active[table_eq_], 0);
  plugin.cleanup();
}
