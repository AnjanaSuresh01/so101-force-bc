"""griff -- contact-rich behaviour cloning on an SO-101 arm.

Layout:

    griff.kinematics  forward kinematics, Jacobians and the 4-DoF IK the SO-101's
                      geometry actually admits
    griff.sensing     contact-force estimation from servo load (no F/T sensor)
    griff.control     the force-limited admittance controller
    griff.sim         MuJoCo task environments
    griff.teleop      leader-follower rig, Feetech bus driver, episode recorder
    griff.data        LeRobot v2.1 dataset writer, reader and validator
    griff.policies    ACT and Diffusion Policy
    griff.train       behaviour-cloning training loop
    griff.evaluate    rollout harness: task success and peak contact force
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
