from launch import LaunchDescription
from launch_ros.actions import Node
#from launch.substitutions import PathJoinSubstitution
from moveit_configs_utils import MoveItConfigsBuilder
#from launch_ros.substitutions import FindPackageShare
#from ament_index_python.packages import get_package_share_directory
    
def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("ur5")
        .robot_description(file_path="config/ur5.urdf.xacro")
        .robot_description_semantic(file_path="config/ur5.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    move_group_demo = Node(
        name="pickik_arm_control",
        package="data_collection",
        executable="pickik_arm_control",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([move_group_demo])
