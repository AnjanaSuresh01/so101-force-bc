#include "griff_control/force_limited_admittance_controller.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "kdl/tree.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"

namespace griff_control
{

controller_interface::CallbackReturn ForceLimitedAdmittanceController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>("base_link", "base_link");
    auto_declare<std::string>("tool_link", "tool_centre");
    auto_declare<std::string>("policy_command_topic", "~/policy_command");
    auto_declare<std::string>("contact_force_topic", "/contact_force");
    auto_declare<double>("ik_damping", 0.06);
    auto_declare<double>("command_timeout", 0.5);

    auto_declare<double>("admittance.mass", admittance_parameters_.mass);
    auto_declare<double>("admittance.damping", admittance_parameters_.damping);
    auto_declare<double>("admittance.stiffness", admittance_parameters_.stiffness);
    auto_declare<double>("admittance.force_limit", admittance_parameters_.force_limit);
    auto_declare<double>("admittance.deadband", admittance_parameters_.deadband);
    auto_declare<double>("admittance.max_offset", admittance_parameters_.max_offset);
    auto_declare<double>("admittance.max_step", admittance_parameters_.max_step);
    auto_declare<double>("admittance.stiffness_prior", admittance_parameters_.stiffness_prior);
    auto_declare<double>("admittance.stiffness_min", admittance_parameters_.stiffness_min);
    auto_declare<double>("admittance.stiffness_max", admittance_parameters_.stiffness_max);
    auto_declare<double>("admittance.stiffness_smoothing",
      admittance_parameters_.stiffness_smoothing);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_init failed: %s", error.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ForceLimitedAdmittanceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_) {
    configuration.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  return configuration;
}

controller_interface::InterfaceConfiguration
ForceLimitedAdmittanceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_) {
    configuration.names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  return configuration;
}

bool ForceLimitedAdmittanceController::build_chain(const std::string & robot_description)
{
  KDL::Tree tree;
  if (!kdl_parser::treeFromString(robot_description, tree)) {
    RCLCPP_ERROR(get_node()->get_logger(), "could not parse robot_description into a KDL tree");
    return false;
  }
  if (!tree.getChain(base_link_, tool_link_, chain_)) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "no kinematic chain from '%s' to '%s'",
      base_link_.c_str(), tool_link_.c_str());
    return false;
  }
  arm_joints_ = chain_.getNrOfJoints();
  if (arm_joints_ == 0) {
    RCLCPP_ERROR(get_node()->get_logger(), "the chain has no movable joints");
    return false;
  }
  if (arm_joints_ > joint_names_.size()) {
    RCLCPP_ERROR(
      get_node()->get_logger(),
      "the chain has %zu joints but only %zu were configured; the `joints` parameter must list "
      "the chain's joints first, in chain order",
      arm_joints_, joint_names_.size());
    return false;
  }
  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
  jacobian_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(chain_);
  return true;
}

controller_interface::CallbackReturn ForceLimitedAdmittanceController::on_configure(
  const rclcpp_lifecycle::State &)
{
  auto node = get_node();
  joint_names_ = node->get_parameter("joints").as_string_array();
  if (joint_names_.empty()) {
    RCLCPP_ERROR(node->get_logger(), "the `joints` parameter is empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  base_link_ = node->get_parameter("base_link").as_string();
  tool_link_ = node->get_parameter("tool_link").as_string();
  ik_damping_ = node->get_parameter("ik_damping").as_double();
  command_timeout_ = node->get_parameter("command_timeout").as_double();

  admittance_parameters_.dt = 1.0 / std::max(1.0, get_update_rate());
  admittance_parameters_.mass = node->get_parameter("admittance.mass").as_double();
  admittance_parameters_.damping = node->get_parameter("admittance.damping").as_double();
  admittance_parameters_.stiffness = node->get_parameter("admittance.stiffness").as_double();
  admittance_parameters_.force_limit = node->get_parameter("admittance.force_limit").as_double();
  admittance_parameters_.deadband = node->get_parameter("admittance.deadband").as_double();
  admittance_parameters_.max_offset = node->get_parameter("admittance.max_offset").as_double();
  admittance_parameters_.max_step = node->get_parameter("admittance.max_step").as_double();
  admittance_parameters_.stiffness_prior =
    node->get_parameter("admittance.stiffness_prior").as_double();
  admittance_parameters_.stiffness_min =
    node->get_parameter("admittance.stiffness_min").as_double();
  admittance_parameters_.stiffness_max =
    node->get_parameter("admittance.stiffness_max").as_double();
  admittance_parameters_.stiffness_smoothing =
    node->get_parameter("admittance.stiffness_smoothing").as_double();

  try {
    admittance_parameters_.validate();
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "admittance parameters rejected: %s", error.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  admittance_ = std::make_unique<AdmittanceCore>(admittance_parameters_);

  if (!build_chain(get_robot_description())) {
    return controller_interface::CallbackReturn::ERROR;
  }

  lower_limits_.assign(joint_names_.size(), -std::numeric_limits<double>::infinity());
  upper_limits_.assign(joint_names_.size(), std::numeric_limits<double>::infinity());
  for (std::size_t index = 0, segment = 0; segment < chain_.getNrOfSegments(); ++segment) {
    const auto & joint = chain_.getSegment(segment).getJoint();
    if (joint.getType() == KDL::Joint::None) {continue;}
    lower_limits_[index] = -std::numeric_limits<double>::infinity();
    upper_limits_[index] = std::numeric_limits<double>::infinity();
    ++index;
  }

  measured_positions_ = Eigen::VectorXd::Zero(joint_names_.size());
  commanded_positions_ = Eigen::VectorXd::Zero(joint_names_.size());

  policy_subscription_ = node->create_subscription<sensor_msgs::msg::JointState>(
    node->get_parameter("policy_command_topic").as_string(), rclcpp::SystemDefaultsQoS(),
    [this](const std::shared_ptr<sensor_msgs::msg::JointState> message) {
      policy_command_.writeFromNonRT(message);
    });
  force_subscription_ = node->create_subscription<griff_msgs::msg::ContactForce>(
    node->get_parameter("contact_force_topic").as_string(), rclcpp::SystemDefaultsQoS(),
    [this](const std::shared_ptr<griff_msgs::msg::ContactForce> message) {
      contact_force_.writeFromNonRT(message);
    });

  status_publisher_base_ =
    node->create_publisher<griff_msgs::msg::GuardStatus>("~/status", rclcpp::SystemDefaultsQoS());
  status_publisher_ =
    std::make_unique<realtime_tools::RealtimePublisher<griff_msgs::msg::GuardStatus>>(
    status_publisher_base_);

  RCLCPP_INFO(
    node->get_logger(),
    "force-limited admittance ready: %zu joints, chain %s -> %s (%zu dof), limit %.2f N at %.0f Hz",
    joint_names_.size(), base_link_.c_str(), tool_link_.c_str(), arm_joints_,
    admittance_parameters_.force_limit, get_update_rate());
  return controller_interface::CallbackReturn::SUCCESS;
}

void ForceLimitedAdmittanceController::read_state()
{
  for (std::size_t index = 0; index < joint_names_.size(); ++index) {
    measured_positions_(static_cast<Eigen::Index>(index)) =
      state_interfaces_[index].get_optional().value_or(
      measured_positions_(static_cast<Eigen::Index>(index)));
  }
}

controller_interface::CallbackReturn ForceLimitedAdmittanceController::on_activate(
  const rclcpp_lifecycle::State &)
{
  read_state();
  commanded_positions_ = measured_positions_;
  admittance_->reset(forward_kinematics(measured_positions_));
  have_command_ = false;
  // Hold position until a policy command arrives. Activating into whatever the
  // last publisher left in the buffer is how a controller lunges on startup.
  policy_command_.writeFromNonRT(std::shared_ptr<sensor_msgs::msg::JointState>());
  contact_force_.writeFromNonRT(std::shared_ptr<griff_msgs::msg::ContactForce>());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ForceLimitedAdmittanceController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

Eigen::Vector3d ForceLimitedAdmittanceController::forward_kinematics(
  const Eigen::VectorXd & joints) const
{
  KDL::JntArray positions(static_cast<unsigned int>(arm_joints_));
  for (std::size_t index = 0; index < arm_joints_; ++index) {
    positions(static_cast<unsigned int>(index)) = joints(static_cast<Eigen::Index>(index));
  }
  KDL::Frame frame;
  fk_solver_->JntToCart(positions, frame);
  return Eigen::Vector3d(frame.p.x(), frame.p.y(), frame.p.z());
}

Eigen::Vector3d ForceLimitedAdmittanceController::tool_axis(
  const Eigen::VectorXd & joints) const
{
  // The direction the tool points, and therefore the one direction in which it
  // can press something. The admittance is allowed to yield along this and
  // nowhere else -- yielding sideways means yielding to friction.
  KDL::JntArray positions(static_cast<unsigned int>(arm_joints_));
  for (std::size_t index = 0; index < arm_joints_; ++index) {
    positions(static_cast<unsigned int>(index)) = joints(static_cast<Eigen::Index>(index));
  }
  KDL::Frame frame;
  fk_solver_->JntToCart(positions, frame);
  return Eigen::Vector3d(frame.M(0, 2), frame.M(1, 2), frame.M(2, 2));
}

Eigen::VectorXd ForceLimitedAdmittanceController::cartesian_to_joint_delta(
  const Eigen::VectorXd & joints, const Eigen::Vector3d & delta) const
{
  KDL::JntArray positions(static_cast<unsigned int>(arm_joints_));
  for (std::size_t index = 0; index < arm_joints_; ++index) {
    positions(static_cast<unsigned int>(index)) = joints(static_cast<Eigen::Index>(index));
  }
  KDL::Jacobian jacobian(static_cast<unsigned int>(arm_joints_));
  jacobian_solver_->JntToJac(positions, jacobian);

  Eigen::MatrixXd linear(3, static_cast<Eigen::Index>(arm_joints_));
  for (std::size_t column = 0; column < arm_joints_; ++column) {
    for (int row = 0; row < 3; ++row) {
      linear(row, static_cast<Eigen::Index>(column)) =
        jacobian(static_cast<unsigned int>(row), static_cast<unsigned int>(column));
    }
  }

  // Damped least squares. The damping is not small, and that is deliberate:
  // this arm passes through a shoulder singularity at the reach where the
  // fixtures sit, and an undamped pseudo-inverse there produces joint steps
  // large enough to make the follower snap.
  const Eigen::Matrix3d gram =
    linear * linear.transpose() + (ik_damping_ * ik_damping_) * Eigen::Matrix3d::Identity();
  return linear.transpose() * gram.ldlt().solve(delta);
}

controller_interface::return_type ForceLimitedAdmittanceController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  read_state();

  const auto command = *policy_command_.readFromRT();
  const auto force_message = *contact_force_.readFromRT();

  Eigen::VectorXd requested = commanded_positions_;
  if (command && command->position.size() >= joint_names_.size()) {
    for (std::size_t index = 0; index < joint_names_.size(); ++index) {
      requested(static_cast<Eigen::Index>(index)) = command->position[index];
    }
    last_command_stamp_ = time;
    have_command_ = true;
  } else if (!have_command_) {
    // Nothing has ever commanded us. Hold the measured pose.
    requested = measured_positions_;
  }

  // A policy that has stopped publishing is a policy that has crashed. Freezing
  // at the last command is the only safe reading of that, and it must not be
  // silent.
  if (have_command_ && (time - last_command_stamp_).seconds() > command_timeout_) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "no policy command for %.2f s; holding position",
      (time - last_command_stamp_).seconds());
    requested = commanded_positions_;
  }

  Eigen::Vector3d force = Eigen::Vector3d::Zero();
  if (force_message) {
    force = Eigen::Vector3d(
      force_message->force.x, force_message->force.y, force_message->force.z);
  }

  const Eigen::Vector3d target = forward_kinematics(requested);
  const Eigen::Vector3d axis = tool_axis(requested);
  const Eigen::Vector3d reference = admittance_->step(target, force, &axis);
  const Eigen::VectorXd joint_delta = cartesian_to_joint_delta(requested, reference - target);

  commanded_positions_ = requested;
  for (std::size_t index = 0; index < arm_joints_; ++index) {
    const auto i = static_cast<Eigen::Index>(index);
    commanded_positions_(i) = std::clamp(
      requested(i) + joint_delta(i), lower_limits_[index], upper_limits_[index]);
  }

  for (std::size_t index = 0; index < joint_names_.size(); ++index) {
    if (!command_interfaces_[index].set_value(
        commanded_positions_(static_cast<Eigen::Index>(index))))
    {
      RCLCPP_ERROR_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "could not write the command for joint '%s'", joint_names_[index].c_str());
      return controller_interface::return_type::ERROR;
    }
  }

  if (status_publisher_ && status_publisher_->trylock()) {
    auto & message = status_publisher_->msg_;
    message.header.stamp = time;
    message.header.frame_id = base_link_;
    message.governed = admittance_->governed();
    message.force_limit = admittance_parameters_.force_limit;
    message.measured_force = force.norm();
    message.environment_stiffness = admittance_->environment_stiffness();
    message.correction = (reference - target).norm();
    message.governed_cycles = admittance_->governed_cycles();
    status_publisher_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

}  // namespace griff_control

PLUGINLIB_EXPORT_CLASS(
  griff_control::ForceLimitedAdmittanceController, controller_interface::ControllerInterface)
