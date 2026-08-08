"""Publishes a contact-force estimate from what the servos report.

Subscribes to /joint_states, whose `effort` field carries STS3215 present-load
converted to N.m by the hardware interface, and publishes griff_msgs/ContactForce
at the rate joint states arrive. The estimation itself is `griff.sensing`, the
same code the policies were trained and evaluated against -- the node is a
transport wrapper and deliberately contains no maths of its own, so there is one
implementation of the force estimate rather than two that drift.

NOT EXECUTED. Builds in CI; has never run against a robot or a bag.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from griff_msgs.msg import ContactForce

try:  # pragma: no cover - exercised only in a sourced ROS environment
    import mujoco

    from griff.kinematics import JOINT_NAMES, position_jacobian
    from griff.paths import CALIBRATION, scene
    from griff.sensing import ContactForceEstimator, ForceCalibration
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "griff_policy needs the `griff` package importable from the ROS Python "
        "environment: pip install -e /path/to/so101-force-bc"
    ) from error


class ForceEstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("griff_force_estimator")
        self.declare_parameter("calibration", str(CALIBRATION / "peg_insert.json"))
        self.declare_parameter("scene", "peg_insert")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("smoothing", 0.4)

        calibration_path = self.get_parameter("calibration").value
        self.frame_id = self.get_parameter("frame_id").value
        self.estimator = ContactForceEstimator(
            ForceCalibration.load(calibration_path),
            smoothing=float(self.get_parameter("smoothing").value),
        )

        # The Jacobian is evaluated on the same kinematic model the estimator was
        # calibrated against. Using MuJoCo here rather than KDL is not laziness:
        # a Jacobian from a slightly different model produces a force estimate
        # that is wrong in a way no test in this repo would catch.
        self.model = mujoco.MjModel.from_xml_path(str(scene(self.get_parameter("scene").value)))
        self.data = mujoco.MjData(self.model)

        self.publisher = self.create_publisher(ContactForce, "/griff/contact_force", 10)
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.on_joint_state, 10
        )
        self.get_logger().info(
            f"contact-force estimator up: calibration {calibration_path}, "
            f"fit residual {np.round(self.estimator.calibration.residual_rms, 4).tolist()} N.m"
        )

    def on_joint_state(self, message: JointState) -> None:
        if not message.effort:
            self.get_logger().warn(
                "joint_states carries no effort field; the SO-101 hardware interface must "
                "publish STS3215 present-load as effort or there is nothing to estimate from",
                throttle_duration_sec=5.0,
            )
            return

        index = {name: i for i, name in enumerate(message.name)}
        try:
            order = [index[name] for name in JOINT_NAMES]
        except KeyError as missing:
            self.get_logger().warn(
                f"joint_states is missing {missing}; expected {list(JOINT_NAMES)}",
                throttle_duration_sec=5.0,
            )
            return

        q = np.array([message.position[i] for i in order])
        qd = (
            np.array([message.velocity[i] for i in order])
            if len(message.velocity) == len(message.name)
            else np.zeros(6)
        )
        tau = np.array([message.effort[i] for i in order])

        self.data.qpos[:6] = q
        self.data.qvel[:6] = qd
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        estimate = self.estimator.estimate(q, qd, tau, position_jacobian(self.model, self.data))

        out = ContactForce()
        out.header.stamp = message.header.stamp
        out.header.frame_id = self.frame_id
        out.force.x, out.force.y, out.force.z = (float(v) for v in estimate.force)
        out.magnitude = estimate.magnitude
        out.condition_number = estimate.condition_number
        out.residual_torque = [float(v) for v in estimate.residual_torque]
        self.publisher.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForceEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
