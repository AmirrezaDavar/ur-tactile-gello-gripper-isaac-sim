#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

class PickIkExampleNode : public rclcpp::Node
{
public:
  PickIkExampleNode() : Node("pickik_example_node")
  {
    // Constructor left empty intentionally
  }

  void init()
  {
    // Use rclcpp::Node’s own shared_from_this()
    robot_model_loader_ = std::make_shared<robot_model_loader::RobotModelLoader>(
        this->shared_from_this(), "robot_description");

    robot_model_ = robot_model_loader_->getModel();
    if (!robot_model_)
    {
      RCLCPP_ERROR(get_logger(), "Failed to load robot model");
      return;
    }

    robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    robot_state_->setToDefaultValues();

    jmg_ = robot_model_->getJointModelGroup("arm");
    if (!jmg_)
    {
      RCLCPP_ERROR(get_logger(), "Failed to get joint model group 'arm'");
      return;
    }

    // Subscribers and publishers
    pose_sub_ = create_subscription<geometry_msgs::msg::Pose>(
        "/target_pose", 10,
        std::bind(&PickIkExampleNode::poseCallback, this, std::placeholders::_1));

    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

    RCLCPP_INFO(get_logger(), "PickIK node initialized successfully.");
  }

private:
  void poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg)
  {
    double timeout = 0.1;
    bool found_ik = robot_state_->setFromIK(jmg_, *msg, timeout);

    if (found_ik)
    {
      std::vector<double> joint_values;
      robot_state_->copyJointGroupPositions(jmg_, joint_values);

      RCLCPP_INFO(get_logger(), "Found IK solution:");
      sensor_msgs::msg::JointState joint_msg;
      joint_msg.header.stamp = now();

      for (size_t i = 0; i < joint_values.size(); ++i)
      {
        RCLCPP_INFO(get_logger(), "  joint[%zu] = %f", i, joint_values[i]);
        joint_msg.name.push_back(jmg_->getVariableNames()[i]);
        joint_msg.position.push_back(joint_values[i]);
      }

      joint_pub_->publish(joint_msg);
    }
    else
    {
      RCLCPP_WARN(get_logger(), "IK solution not found");
    }
  }

  // MoveIt-related members
  std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader_;
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  const moveit::core::JointModelGroup* jmg_{nullptr};

  // ROS 2 interfaces
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PickIkExampleNode>();
  node->init();  // call after construction
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
