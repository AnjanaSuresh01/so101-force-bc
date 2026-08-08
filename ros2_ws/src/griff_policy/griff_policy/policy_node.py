"""Runs a trained ACT or Diffusion Policy checkpoint at the control rate.

Subscribes to the two camera topics, /joint_states and the contact-force
estimate; publishes the joint positions the policy would like on
/griff/policy_command. It does NOT command the arm. What the arm is commanded is
decided by griff_control's force-limited admittance controller, which takes this
topic as a request -- the separation is the point, and it is why this node has
no command interface of its own to get wrong.

Two safety behaviours that matter more than they look:

* **Observations must be fresh and complete.** If any of the four inputs is
  older than `observation_timeout`, the node publishes nothing rather than
  feeding the policy a stale image. A behaviour-cloning policy given a frozen
  camera will happily keep acting on it.
* **The gripper is passed through unchanged.** These tasks pre-grasp the tool,
  and a policy that has learned a constant gripper value is not a reason to let
  it drive one.

NOT EXECUTED. Builds in CI; has never run against a robot.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState

from griff_msgs.msg import ContactForce

try:  # pragma: no cover - exercised only in a sourced ROS environment
    from griff.kinematics import JOINT_NAMES
    from griff.policies.runner import load_policy
    from griff.sim.env import Observation
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "griff_policy needs the `griff` package importable from the ROS Python "
        "environment: pip install -e /path/to/so101-force-bc"
    ) from error

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def image_to_array(message: Image, size: int) -> np.ndarray:
    """Decode an Image to HxWx3 uint8 without pulling in cv_bridge.

    Only rgb8 and bgr8 are accepted. Silently reinterpreting some other encoding
    would hand the policy an image whose channels are in an order it never saw
    in training, and it would look like a policy failure rather than a bug.
    """
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {message.encoding!r}; want rgb8 or bgr8")
    array = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3)
    if message.encoding == "bgr8":
        array = array[:, :, ::-1]
    if (message.height, message.width) != (size, size):
        raise ValueError(
            f"expected {size}x{size} images to match the policy's input, got "
            f"{message.height}x{message.width}; resize upstream so the resampling "
            "is explicit and identical to what recording did"
        )
    return np.ascontiguousarray(array)


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("griff_policy")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("observation_timeout", 0.25)
        self.declare_parameter("top_image_topic", "/cameras/top/image_raw")
        self.declare_parameter("wrist_image_topic", "/cameras/wrist/image_raw")

        checkpoint = self.get_parameter("checkpoint").value
        if not checkpoint:
            raise ValueError("the `checkpoint` parameter is required")
        self.runner = load_policy(checkpoint)
        self.runner.reset()
        self.config = self.runner.config
        self.timeout = float(self.get_parameter("observation_timeout").value)

        self._images: dict[str, np.ndarray] = {}
        self._stamps: dict[str, float] = {}
        self._state: np.ndarray | None = None
        self._force = np.zeros(3)

        self.create_subscription(
            Image, self.get_parameter("top_image_topic").value,
            lambda m: self.on_image("top", m), SENSOR_QOS,
        )
        self.create_subscription(
            Image, self.get_parameter("wrist_image_topic").value,
            lambda m: self.on_image("wrist", m), SENSOR_QOS,
        )
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        self.create_subscription(ContactForce, "/griff/contact_force", self.on_force, 10)
        self.publisher = self.create_publisher(JointState, "/griff/policy_command", 10)

        rate = float(self.get_parameter("rate").value)
        self.timer = self.create_timer(1.0 / rate, self.on_tick)
        self.get_logger().info(
            f"policy up: {self.config.kind}/{self.config.conditioning} from {checkpoint}, "
            f"{rate:.0f} Hz, cameras {list(self.config.cameras)}"
        )
        if not self.config.uses_force:
            self.get_logger().warn(
                "this checkpoint was trained without the force channel; the contact-force "
                "subscription is still needed by the admittance controller, but the policy "
                "itself cannot see contact"
            )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, camera: str, message: Image) -> None:
        try:
            self._images[camera] = image_to_array(message, self.config.image_size)
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=5.0)
            return
        self._stamps[camera] = self._now()

    def on_joint_state(self, message: JointState) -> None:
        index = {name: i for i, name in enumerate(message.name)}
        if not all(name in index for name in JOINT_NAMES):
            return
        self._state = np.array([message.position[index[name]] for name in JOINT_NAMES])
        self._stamps["state"] = self._now()

    def on_force(self, message: ContactForce) -> None:
        self._force = np.array([message.force.x, message.force.y, message.force.z])
        self._stamps["force"] = self._now()

    def _stale(self) -> list[str]:
        required = [*self.config.cameras, "state"]
        if self.config.uses_force:
            required.append("force")
        now = self._now()
        return [key for key in required if now - self._stamps.get(key, -1e9) > self.timeout]

    def on_tick(self) -> None:
        stale = self._stale()
        if stale or self._state is None:
            self.get_logger().warn(
                f"not acting: stale or missing inputs {sorted(stale)}",
                throttle_duration_sec=2.0,
            )
            return

        observation = Observation(
            state=self._state.astype(np.float32),
            force=self._force.astype(np.float32),
            images={camera: self._images[camera] for camera in self.config.cameras},
        )
        action = self.runner.act(observation)

        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = [float(v) for v in action[:5]] + [float(self._state[5])]
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
