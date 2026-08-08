"""The URDF and the MJCF have to describe the same robot.

The MuJoCo model is what the policies were trained and evaluated against. The
URDF is what MoveIt plans with and what the ROS admittance controller computes
forward kinematics from. If the two drift apart, the controller's idea of where
the tool is stops matching the robot the policy learned on, and the force limit
silently becomes a limit on the wrong point in space -- which is the kind of bug
that is only found by damaging something.

Parsed as XML rather than loaded through ROS, so this runs in the same CI job as
everything else and does not need a sourced workspace.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from pathlib import Path

import mujoco
import numpy as np
import pytest

from griff.kinematics import JOINT_NAMES
from griff.paths import ASSETS, REPO_ROOT

URDF = REPO_ROOT / "ros2_ws" / "src" / "griff_description" / "urdf" / "so101.urdf.xacro"
SRDF = REPO_ROOT / "ros2_ws" / "src" / "griff_description" / "urdf" / "so101.srdf"

#: xacro properties are simple `${name}` substitutions here; resolving them with
#: a regex rather than running xacro keeps this test free of a ROS dependency.
_PROPERTY = "{http://www.ros.org/wiki/xacro}property"


def _properties(root: ElementTree.Element) -> dict[str, float]:
    values = {}
    for element in root.iter(_PROPERTY):
        try:
            values[element.attrib["name"]] = float(element.attrib["value"])
        except (KeyError, ValueError):
            continue
    return values


def _resolve(text: str, properties: dict[str, float]) -> float:
    text = text.strip()
    if text.startswith("${") and text.endswith("}"):
        return properties[text[2:-1].strip()]
    return float(text)


@pytest.fixture(scope="module")
def urdf() -> tuple[ElementTree.Element, dict[str, float]]:
    root = ElementTree.parse(URDF).getroot()
    return root, _properties(root)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(ASSETS / "task_free.xml"))


def _urdf_joints(root: ElementTree.Element) -> dict[str, ElementTree.Element]:
    return {j.attrib["name"]: j for j in root.findall("joint") if "name" in j.attrib}


def test_the_urdf_exists_and_parses(urdf) -> None:
    root, properties = urdf
    assert root.attrib["name"] == "so101"
    assert properties  # xacro properties resolved


def test_both_models_have_the_same_joints(urdf, model) -> None:
    root, _ = urdf
    urdf_joints = _urdf_joints(root)
    movable = {
        name for name, joint in urdf_joints.items() if joint.attrib.get("type") != "fixed"
    }
    mjcf = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)}
    assert movable == mjcf == set(JOINT_NAMES)


def test_joint_limits_agree(urdf, model) -> None:
    root, _ = urdf
    urdf_joints = _urdf_joints(root)
    for name in JOINT_NAMES:
        limit = urdf_joints[name].find("limit")
        assert limit is not None, f"{name} has no <limit>"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[jid]
        assert float(limit.attrib["lower"]) == pytest.approx(low, abs=1e-6), name
        assert float(limit.attrib["upper"]) == pytest.approx(high, abs=1e-6), name


def test_joint_axes_agree(urdf, model) -> None:
    root, _ = urdf
    urdf_joints = _urdf_joints(root)
    for name in JOINT_NAMES:
        axis = urdf_joints[name].find("axis")
        assert axis is not None, f"{name} has no <axis>"
        declared = np.array([float(v) for v in axis.attrib["xyz"].split()])
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert np.allclose(declared, model.jnt_axis[jid], atol=1e-9), name


def test_link_offsets_agree(urdf, model) -> None:
    """The distance from each joint to the next is the arm's geometry.

    Checked against MuJoCo's body positions, which are the same quantity: an
    offset along the parent's z from one joint frame to the next.
    """
    root, properties = urdf
    urdf_joints = _urdf_joints(root)
    expected = {
        "shoulder_pan": "base",
        "shoulder_lift": "shoulder",
        "elbow_flex": "upper_arm",
        "wrist_flex": "forearm",
        "wrist_roll": "wrist",
    }
    for joint_name, mjcf_child in expected.items():
        origin = urdf_joints[joint_name].find("origin")
        offset = np.array([_resolve(v, properties) for v in origin.attrib["xyz"].split()])
        body = {
            "base": "shoulder",
            "shoulder": "upper_arm",
            "upper_arm": "forearm",
            "forearm": "wrist",
            "wrist": "wrist_roll_link",
        }[mjcf_child]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        assert np.allclose(offset, model.body_pos[bid], atol=1e-9), (
            f"{joint_name}: URDF {offset} vs MJCF {model.body_pos[bid]}"
        )


def test_tool_centre_sits_where_the_mjcf_site_does(urdf, model) -> None:
    """The point the admittance controller regulates and the point the policy
    was trained around must be the same point."""
    root, properties = urdf
    urdf_joints = _urdf_joints(root)
    mount = urdf_joints["tool_mount"].find("origin")
    centre = urdf_joints["tool_centre_joint"].find("origin")
    total = np.array([_resolve(v, properties) for v in mount.attrib["xyz"].split()]) + np.array(
        [_resolve(v, properties) for v in centre.attrib["xyz"].split()]
    )

    tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_center")
    mjcf_total = model.body_pos[tool] + model.site_pos[site]
    assert np.allclose(total, mjcf_total, atol=1e-9)


def test_servo_effort_limits_are_the_sts3215_datasheet_value(urdf) -> None:
    from griff.sensing import SERVO_STALL_TORQUE

    root, properties = urdf
    urdf_joints = _urdf_joints(root)
    for name in JOINT_NAMES[:5]:
        effort = _resolve(urdf_joints[name].find("limit").attrib["effort"], properties)
        assert effort == pytest.approx(SERVO_STALL_TORQUE, abs=1e-6)


def test_ros2_control_exposes_effort_as_a_state_interface(urdf) -> None:
    """The force estimate is built on servo load. Without an effort state
    interface there is no force signal anywhere in the ROS stack."""
    root, _ = urdf
    text = Path(URDF).read_text(encoding="utf-8")
    assert '<state_interface name="effort"/>' in text
    assert root.find("ros2_control") is not None


def test_srdf_planning_group_is_the_five_arm_joints() -> None:
    srdf = ElementTree.parse(SRDF).getroot()
    arm = next(g for g in srdf.findall("group") if g.attrib["name"] == "arm")
    joints = [j.attrib["name"] for j in arm.findall("joint")]
    assert joints == list(JOINT_NAMES[:5])
    assert "gripper" not in joints


def test_every_xml_in_the_repo_is_well_formed() -> None:
    """A "--" inside an XML comment is illegal and strict parsers reject the file.

    MuJoCo's parser is lenient about it and will happily load a scene that
    rosdep, xacro and ElementTree all refuse. Without this check the failure
    surfaces as a ROS build error long after the file was written.
    """
    candidates = [
        *ASSETS.glob("*.xml"),
        *(REPO_ROOT / "ros2_ws").rglob("*.xml"),
        *(REPO_ROOT / "ros2_ws").rglob("*.srdf"),
        *(REPO_ROOT / "ros2_ws").rglob("*.urdf.xacro"),
    ]
    assert candidates
    malformed = []
    for path in candidates:
        try:
            ElementTree.parse(path)
        except ElementTree.ParseError as error:
            malformed.append(f"{path.relative_to(REPO_ROOT)}: {error}")
    assert not malformed, "; ".join(malformed)
