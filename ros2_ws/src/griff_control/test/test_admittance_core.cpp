// The same assertions as tests/test_admittance.py, against the same wall model.
//
// The point is not that C++ arithmetic works. It is that the controller which
// would run on the robot and the controller the published results were produced
// with enforce the same bound. If these two ever disagree, the numbers in
// results/RESULTS.md stop describing the thing that would be deployed.

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "griff_control/admittance_core.hpp"

namespace
{

struct Trace
{
  std::vector<double> force;
  std::vector<double> reference;
};

/// Position-controlled arm meeting a linear spring wall at z = 0. The policy
/// ramps its target 50 mm *below* the wall and holds -- i.e. asks for something
/// ruinous -- and the arm tracks the commanded reference exactly, which is the
/// worst case a position-commanded robot can face.
Trace press(
  griff_control::AdmittanceCore & core, double environment_stiffness,
  double start_z = 0.030, double commanded_depth = 0.050, int cycles = 400)
{
  core.reset(Eigen::Vector3d(0.0, 0.0, start_z));
  Trace trace;
  Eigen::Vector3d force = Eigen::Vector3d::Zero();
  const int ramp = cycles / 2;
  for (int i = 0; i < cycles; ++i) {
    const double alpha = std::min(1.0, static_cast<double>(i) / static_cast<double>(ramp - 1));
    const double z = start_z + alpha * (-commanded_depth - start_z);
    const Eigen::Vector3d reference = core.step(Eigen::Vector3d(0.0, 0.0, z), force);
    const double penetration = std::max(0.0, -reference.z());
    force = Eigen::Vector3d(0.0, 0.0, environment_stiffness * penetration);
    trace.force.push_back(force.norm());
    trace.reference.push_back(reference.z());
  }
  return trace;
}

double settled_max(const Trace & trace, std::size_t tail = 60)
{
  const std::size_t begin = trace.force.size() > tail ? trace.force.size() - tail : 0;
  return *std::max_element(trace.force.begin() + static_cast<long>(begin), trace.force.end());
}

TEST(AdmittanceCore, FreeSpaceIsTransparent)
{
  griff_control::AdmittanceCore core;
  core.reset(Eigen::Vector3d(0.0, 0.0, 0.10));
  const Eigen::Vector3d target(0.02, -0.01, 0.08);
  Eigen::Vector3d reference;
  for (int i = 0; i < 200; ++i) {
    reference = core.step(target, Eigen::Vector3d::Zero());
  }
  EXPECT_NEAR((reference - target).norm(), 0.0, 1e-6);
  EXPECT_EQ(core.governed_cycles(), 0u);
}

TEST(AdmittanceCore, SteadyStateForceIsBounded)
{
  for (const double stiffness : {400.0, 1000.0, 2500.0, 8000.0, 25000.0}) {
    griff_control::AdmittanceCore core;
    const auto trace = press(core, stiffness);
    const auto & p = core.parameters();
    EXPECT_LE(settled_max(trace), p.force_limit + p.deadband)
      << "environment stiffness " << stiffness << " N/m";
  }
}

TEST(AdmittanceCore, TransientStaysNearTheLimit)
{
  for (const double stiffness : {400.0, 1000.0, 2500.0, 8000.0}) {
    griff_control::AdmittanceCore core;
    const auto trace = press(core, stiffness);
    const double peak = *std::max_element(trace.force.begin(), trace.force.end());
    EXPECT_LE(peak, 1.5 * core.parameters().force_limit)
      << "environment stiffness " << stiffness << " N/m";
  }
}

TEST(AdmittanceCore, GovernorNeverAdvancesIntoAnOverLimitContact)
{
  griff_control::AdmittanceCore core;
  core.reset(Eigen::Vector3d::Zero());
  const Eigen::Vector3d over_limit(0.0, 0.0, core.parameters().force_limit * 3.0);
  Eigen::Vector3d previous = Eigen::Vector3d::Zero();
  for (int i = 0; i < 50; ++i) {
    const Eigen::Vector3d reference = core.step(Eigen::Vector3d(0.0, 0.0, -0.10), over_limit);
    EXPECT_GE(reference.z(), previous.z() - 1e-9);
    previous = reference;
  }
  EXPECT_TRUE(core.governed());
}

TEST(AdmittanceCore, LowerLimitYieldsLowerForce)
{
  std::vector<double> peaks;
  for (const double limit : {3.0, 6.0, 12.0}) {
    griff_control::AdmittanceParameters parameters;
    parameters.force_limit = limit;
    griff_control::AdmittanceCore core(parameters);
    peaks.push_back(settled_max(press(core, 2500.0)));
  }
  EXPECT_LT(peaks[0], peaks[1]);
  EXPECT_LT(peaks[1], peaks[2]);
}

TEST(AdmittanceCore, StiffnessEstimateConvergesOnTheTrueWall)
{
  griff_control::AdmittanceCore core;
  press(core, 3000.0);
  EXPECT_GT(core.environment_stiffness(), 1500.0);
  EXPECT_LT(core.environment_stiffness(), 6000.0);
}

TEST(AdmittanceCore, ContactReleaseReturnsTheOffsetToZero)
{
  griff_control::AdmittanceCore core;
  press(core, 2500.0, 0.030, 0.050, 200);
  EXPECT_GT(core.offset().norm(), 1e-3);
  const Eigen::Vector3d target(0.0, 0.0, 0.05);
  for (int i = 0; i < 300; ++i) {
    core.step(target, Eigen::Vector3d::Zero());
  }
  EXPECT_LT(core.offset().norm(), 1e-4);
}

TEST(AdmittanceCore, ComplianceAxisConfinesTheOffsetToThatAxis)
{
  // Three-axis compliance also yields to friction, which opposes motion. The
  // Python suite asserts the same property; see the note on step().
  const Eigen::Vector3d axis(0.0, 0.0, 1.0);
  const Eigen::Vector3d tangential(4.0, 0.0, 0.0);

  griff_control::AdmittanceCore confined;
  confined.reset(Eigen::Vector3d::Zero());
  for (int i = 0; i < 120; ++i) {
    confined.step(Eigen::Vector3d::Zero(), tangential, &axis);
  }
  EXPECT_LT(confined.offset().head<2>().norm(), 1e-9);

  griff_control::AdmittanceCore unconfined;
  unconfined.reset(Eigen::Vector3d::Zero());
  for (int i = 0; i < 120; ++i) {
    unconfined.step(Eigen::Vector3d::Zero(), tangential);
  }
  EXPECT_GT(unconfined.offset().head<2>().norm(), 1e-3);
}

TEST(AdmittanceCore, BoundDoesNotDependOnTheComplianceStiffness)
{
  // Tasks that must sustain force are given a stiff virtual spring, so it
  // matters that raising it does not quietly raise the settled force.
  for (const double stiffness : {250.0, 900.0, 2200.0, 6000.0}) {
    griff_control::AdmittanceParameters parameters;
    parameters.stiffness = stiffness;
    griff_control::AdmittanceCore core(parameters);
    const auto trace = press(core, 2500.0);
    EXPECT_LE(settled_max(trace), parameters.force_limit + parameters.deadband)
      << "compliance stiffness " << stiffness << " N/m";
  }
}

TEST(AdmittanceCore, ComplianceAxisMustBeNonZero)
{
  griff_control::AdmittanceCore core;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_THROW(core.step(zero, zero, &zero), std::invalid_argument);
}

TEST(AdmittanceCore, InvalidParametersAreRejected)
{
  griff_control::AdmittanceParameters parameters;
  parameters.force_limit = 0.1;
  parameters.deadband = 0.35;
  EXPECT_THROW(griff_control::AdmittanceCore core(parameters), std::invalid_argument);

  griff_control::AdmittanceParameters negative_mass;
  negative_mass.mass = -1.0;
  EXPECT_THROW(griff_control::AdmittanceCore core(negative_mass), std::invalid_argument);
}

}  // namespace

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
