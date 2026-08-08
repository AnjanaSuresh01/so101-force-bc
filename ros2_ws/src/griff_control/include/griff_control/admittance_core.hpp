// Force-limited admittance, as pure maths.
//
// This is a line-for-line port of griff.control.admittance.AdmittanceController
// (Python), and the duplication is deliberate. The Python version is what the
// policies were evaluated against and what produced the numbers in
// results/RESULTS.md; this version is what would run on the robot. Two
// implementations of a safety bound that are supposed to agree are worth having
// only if their agreement is checked, so test/test_admittance_core.cpp asserts
// the same properties the Python test suite asserts, against the same
// spring-wall model and the same numbers.
//
// Header-only so the maths can be unit tested without spinning a controller
// manager, a robot description, or a node.

#ifndef GRIFF_CONTROL__ADMITTANCE_CORE_HPP_
#define GRIFF_CONTROL__ADMITTANCE_CORE_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace griff_control
{

struct AdmittanceParameters
{
  double dt = 1.0 / 30.0;
  double mass = 1.2;              // kg, virtual
  double damping = 45.0;          // N.s/m
  double stiffness = 250.0;       // N/m, pulls the offset back to zero
  double force_limit = 8.0;       // N, the bound the governor enforces
  double deadband = 0.35;         // N, below this the estimate reads as noise
  double max_offset = 0.035;      // m, compliance authority
  double max_step = 0.006;        // m per cycle, free-space slew limit
  double stiffness_prior = 4000.0;      // N/m
  double stiffness_min = 250.0;         // N/m
  double stiffness_max = 80000.0;       // N/m
  double stiffness_smoothing = 0.35;
  double min_probe_distance = 5e-5;     // m

  void validate() const
  {
    if (dt <= 0.0) {throw std::invalid_argument("dt must be positive");}
    if (mass <= 0.0) {throw std::invalid_argument("mass must be positive");}
    if (damping < 0.0 || stiffness < 0.0) {
      throw std::invalid_argument("damping and stiffness must be non-negative");
    }
    if (force_limit <= deadband) {
      throw std::invalid_argument("force_limit must exceed the deadband");
    }
    if (max_offset <= 0.0 || max_step <= 0.0) {
      throw std::invalid_argument("max_offset and max_step must be positive");
    }
    if (!(stiffness_min > 0.0 && stiffness_min < stiffness_max)) {
      throw std::invalid_argument("stiffness bounds must be an increasing positive pair");
    }
    if (stiffness_prior < stiffness_min || stiffness_prior > stiffness_max) {
      throw std::invalid_argument("stiffness_prior must lie inside the stiffness bounds");
    }
    if (!(stiffness_smoothing > 0.0 && stiffness_smoothing <= 1.0)) {
      throw std::invalid_argument("stiffness_smoothing must be in (0, 1]");
    }
  }
};

class AdmittanceCore
{
public:
  explicit AdmittanceCore(const AdmittanceParameters & parameters = AdmittanceParameters())
  : parameters_(parameters)
  {
    parameters_.validate();
    reset();
  }

  void reset()
  {
    offset_.setZero();
    velocity_.setZero();
    last_delta_.setZero();
    last_direction_.setZero();
    has_direction_ = false;
    has_reference_ = false;
    reference_.setZero();
    stiffness_ = parameters_.stiffness_prior;
    last_magnitude_ = 0.0;
    probes_ = 0;
    governed_ = false;
    governed_cycles_ = 0;
    peak_force_ = 0.0;
  }

  void reset(const Eigen::Vector3d & reference)
  {
    reset();
    reference_ = reference;
    has_reference_ = true;
  }

  /// One control cycle. `force` is what the environment exerts on the tool,
  /// in the same frame as `policy_target`.
  Eigen::Vector3d step(const Eigen::Vector3d & policy_target, const Eigen::Vector3d & force)
  {
    const auto & p = parameters_;
    const double magnitude = force.norm();
    peak_force_ = std::max(peak_force_, magnitude);

    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    bool have_direction = false;
    if (magnitude > p.deadband) {
      direction = force / magnitude;
      have_direction = true;
    } else if (has_direction_) {
      direction = last_direction_;
      have_direction = true;
    }
    update_stiffness(magnitude, direction, have_direction);

    Eigen::Vector3d effective = Eigen::Vector3d::Zero();
    if (magnitude > p.deadband) {
      effective = force * ((magnitude - p.deadband) / magnitude);
    }

    // Backward Euler on  M x'' + D x' + K x = -F, solved for the new velocity.
    // Unconditionally stable, which matters at a 33 ms cycle against a rigid
    // fixture -- explicit integration there diverges.
    const double denominator = p.mass + p.dt * p.damping + p.dt * p.dt * p.stiffness;
    velocity_ = (p.mass * velocity_ - p.dt * (effective + p.stiffness * offset_)) / denominator;
    Eigen::Vector3d offset = offset_ + p.dt * velocity_;

    const double offset_norm = offset.norm();
    if (offset_norm > p.max_offset) {
      offset *= p.max_offset / offset_norm;
      const Eigen::Vector3d radial = offset / std::max(offset.norm(), 1e-12);
      const double outward = velocity_.dot(radial);
      if (outward > 0.0) {
        velocity_ -= outward * radial;
      }
    }
    offset_ = offset;

    const Eigen::Vector3d reference = policy_target + offset_;
    const Eigen::Vector3d previous = has_reference_ ? reference_ : reference;

    Eigen::Vector3d delta = reference - previous;
    const double step_norm = delta.norm();
    if (step_norm > p.max_step) {
      delta *= p.max_step / step_norm;
    }

    governed_ = false;
    if (magnitude > p.deadband && have_direction) {
      const double advance = -delta.dot(direction);
      const double headroom = p.force_limit - magnitude;
      const double allowance = std::max(headroom / std::max(stiffness_, 1e-12), -p.max_step);
      if (advance > allowance) {
        delta += (advance - allowance) * direction;
        governed_ = true;
        ++governed_cycles_;
      }
      last_direction_ = direction;
      has_direction_ = true;
    }

    last_delta_ = delta;
    last_magnitude_ = magnitude;
    reference_ = previous + delta;
    has_reference_ = true;
    return reference_;
  }

  bool governed() const {return governed_;}
  std::uint64_t governed_cycles() const {return governed_cycles_;}
  double environment_stiffness() const {return stiffness_;}
  double peak_force() const {return peak_force_;}
  const Eigen::Vector3d & offset() const {return offset_;}
  const AdmittanceParameters & parameters() const {return parameters_;}

private:
  void update_stiffness(double magnitude, const Eigen::Vector3d & direction, bool have_direction)
  {
    if (!have_direction) {return;}
    const double advance = -last_delta_.dot(direction);
    if (std::abs(advance) < parameters_.min_probe_distance) {return;}
    const double rise = magnitude - last_magnitude_;
    if (rise * advance <= 0.0) {return;}

    const double observed = rise / advance;
    // First observation replaces the prior outright: the prior is a guess and
    // the first real contact is data.
    const double alpha = (probes_ == 0) ? 1.0 : parameters_.stiffness_smoothing;
    ++probes_;
    const double blended = (1.0 - alpha) * stiffness_ + alpha * observed;
    stiffness_ = std::clamp(blended, parameters_.stiffness_min, parameters_.stiffness_max);
  }

  AdmittanceParameters parameters_;
  Eigen::Vector3d offset_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d reference_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d last_delta_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d last_direction_{Eigen::Vector3d::Zero()};
  bool has_reference_{false};
  bool has_direction_{false};
  double stiffness_{4000.0};
  double last_magnitude_{0.0};
  double peak_force_{0.0};
  std::uint64_t probes_{0};
  std::uint64_t governed_cycles_{0};
  bool governed_{false};
};

}  // namespace griff_control

#endif  // GRIFF_CONTROL__ADMITTANCE_CORE_HPP_
