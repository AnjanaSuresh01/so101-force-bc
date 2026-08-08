"""Bring up the full stack: robot, controllers, force estimate, policy.

    robot_state_publisher      URDF -> TF
    ros2_control_node          the arm (mock hardware by default)
    joint_state_broadcaster    /joint_states, including servo load as effort
    force_limited_admittance   the guard: /griff/policy_command -> the arm
    griff_policy/force_estimator_node   /joint_states -> /griff/contact_force
    griff_policy/policy_node            cameras + state + force -> a request

Note what is NOT wired: the policy has no path to the arm that does not pass
through the admittance controller. That is the design, and it is why the policy
node publishes a request topic rather than claiming a command interface.

MoveIt Servo is launched only when `use_servo:=true`. It is not on the policy's
path either -- it is there for jogging the arm into place around a policy run,
and it publishes into the same guarded controller.

NOT EXECUTED. This launch file has never been run.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "checkpoint",
            description="Path to a policy.pt produced by `griff train`.",
        ),
        DeclareLaunchArgument(
            "calibration",
            default_value="",
            description="Force-estimator calibration JSON. Defaults to the task's own.",
        ),
        DeclareLaunchArgument("task", default_value="peg_insert"),
        DeclareLaunchArgument("use_mock_hardware", default_value="true"),
        DeclareLaunchArgument("use_servo", default_value="false"),
        DeclareLaunchArgument("rate", default_value="30.0"),
    ]

    robot_description = Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([FindPackageShare("griff_description"), "urdf", "so101.urdf.xacro"]),
        " use_mock_hardware:=", LaunchConfiguration("use_mock_hardware"),
    ])
    controllers = PathJoinSubstitution(
        [FindPackageShare("griff_bringup"), "config", "controllers.yaml"]
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{"robot_description": robot_description}, controllers],
        output="screen",
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )
    admittance_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "force_limited_admittance_controller",
            "--controller-manager", "/controller_manager",
        ],
    )
    force_estimator = Node(
        package="griff_policy",
        executable="force_estimator_node",
        parameters=[{
            "scene": LaunchConfiguration("task"),
            "calibration": LaunchConfiguration("calibration"),
        }],
        output="screen",
    )
    policy = Node(
        package="griff_policy",
        executable="policy_node",
        parameters=[{
            "checkpoint": LaunchConfiguration("checkpoint"),
            "rate": LaunchConfiguration("rate"),
        }],
        output="screen",
    )
    servo = GroupAction(
        condition=IfCondition(LaunchConfiguration("use_servo")),
        actions=[
            Node(
                package="moveit_servo",
                executable="servo_node",
                parameters=[
                    {"robot_description": robot_description},
                    PathJoinSubstitution(
                        [FindPackageShare("griff_bringup"), "config", "servo.yaml"]
                    ),
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        *arguments,
        control_node,
        robot_state_publisher,
        joint_state_broadcaster,
        admittance_controller,
        force_estimator,
        policy,
        servo,
    ])
