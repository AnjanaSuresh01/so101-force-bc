// A ros2_control controller that stands between a learned policy and the arm.
//
// The policy publishes joint positions it would like. This controller decides
// what the arm is actually commanded, by running the policy's request through
// Cartesian admittance and a force-limited reference governor:
//
//   q_policy --FK--> x_policy --admittance+governor--> x_ref
//   dq = J^+ (x_ref - x_policy)          (damped least squares, one step)
//   command = q_policy + dq
//
// Kinematics come from KDL built off the controller's robot_description, so the
// controller has no dependency on a running MoveIt instance -- MoveIt Servo can
// feed it, but it is not required to.
//
// One difference from the simulation guard in griff.control.guard, stated here
// because it is the kind of thing that is otherwise discovered in the lab: the
// Python guard runs a full iterative IK solve to convergence each tick, while
// this runs a single damped-least-squares step. That is the right trade in a
// real-time update() -- an unbounded iteration count has no place in a control
// loop -- but it means large Cartesian corrections are tracked over several
// cycles rather than in one. The force bound is unaffected: it is enforced on
// the Cartesian reference, before any of this.
//
// NEVER RUN ON HARDWARE. This builds in CI against Jazzy; it has not been
// executed against a controller manager or a robot.

#ifndef GRIFF_CONTROL__FORCE_LIMITED_ADMITTANCE_CONTROLLER_HPP_
#define GRIFF_CONTROL__FORCE_LIMITED_ADMITTANCE_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "controller_interface/controller_interface.hpp"
#include "griff_control/admittance_core.hpp"
#include "griff_msgs/msg/contact_force.hpp"
#include "griff_msgs/msg/guard_status.hpp"
#include "kdl/chain.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "kdl/chainjnttojacsolver.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace griff_control
{

class ForceLimitedAdmittanceController : public controller_interface::ControllerInterface
{
public:
  ForceLimitedAdmittanceController() = default;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  bool build_chain(const std::string & robot_description);
  Eigen::Vector3d forward_kinematics(const Eigen::VectorXd & joints) const;
  Eigen::Vector3d tool_axis(const Eigen::VectorXd & joints) const;
  Eigen::VectorXd cartesian_to_joint_delta(
    const Eigen::VectorXd & joints, const Eigen::Vector3d & delta) const;
  void read_state();

  std::vector<std::string> joint_names_;
  std::string base_link_;
  std::string tool_link_;
  double ik_damping_{0.06};
  double command_timeout_{0.5};

  AdmittanceParameters admittance_parameters_;
  std::unique_ptr<AdmittanceCore> admittance_;

  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::unique_ptr<KDL::ChainJntToJacSolver> jacobian_solver_;
  std::size_t arm_joints_{0};  // joints in the KDL chain (the gripper is not one)

  Eigen::VectorXd measured_positions_;
  Eigen::VectorXd commanded_positions_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>> policy_command_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<griff_msgs::msg::ContactForce>> contact_force_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr policy_subscription_;
  rclcpp::Subscription<griff_msgs::msg::ContactForce>::SharedPtr force_subscription_;
  std::unique_ptr<realtime_tools::RealtimePublisher<griff_msgs::msg::GuardStatus>> status_publisher_;
  rclcpp::Publisher<griff_msgs::msg::GuardStatus>::SharedPtr status_publisher_base_;

  rclcpp::Time last_command_stamp_;
  bool have_command_{false};
};

}  // namespace griff_control

#endif  // GRIFF_CONTROL__FORCE_LIMITED_ADMITTANCE_CONTROLLER_HPP_
