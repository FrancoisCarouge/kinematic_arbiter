#include <cmath>
#include <random>

#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/typed_state.hpp"

namespace {

using kinematic_arbiter::core::StateIndex;
namespace typed = kinematic_arbiter::typed;

using StateVector = Eigen::Matrix<double, StateIndex::kFullStateSize, 1>;
using StateMatrix = Eigen::Matrix<double, StateIndex::kFullStateSize, StateIndex::kFullStateSize>;
using PoseMatrix = Eigen::Matrix<double, 6, 6>;

constexpr auto kQw = StateIndex::Quaternion::W;
constexpr auto kPx = StateIndex::Position::X;

PoseMatrix Untyped(const typed::Matrix<typed::Pose, typed::Pose>& pose) {
  return typed::untyped<typed::Matrix<typed::Pose, typed::Pose>>(pose);
}

// At the identity attitude, a rotation is twice the quaternion's vector part:
// its variance four times the quaternion's, its cross-covariance with the
// position twice. The state's quaternion covariance is not the rotation's.
TEST(PoseCovarianceTest, IdentityAttitudeDoublesTheQuaternionVectorPart) {
  StateVector state = StateVector::Zero();
  state(kQw) = 1.0;
  StateMatrix covariance = StateMatrix::Identity() * 1e-2;
  for (int i = 1; i < 4; ++i) covariance(kQw + i, kQw + i) = 1e-4 * i;
  covariance(kPx, kQw + 3) = covariance(kQw + 3, kPx) = 3e-4;

  const PoseMatrix pose = Untyped(typed::pose_covariance(state, covariance));

  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(pose(i, i), 1e-2) << "the position's variance is the state's";
    EXPECT_DOUBLE_EQ(pose(3 + i, 3 + i), 4 * 1e-4 * (i + 1))
        << "the rotation's variance is four times the quaternion component's";
  }
  EXPECT_DOUBLE_EQ(pose(0, 5), 2 * 3e-4) << "the position-rotation cross-covariance";
  EXPECT_DOUBLE_EQ(pose(5, 0), 2 * 3e-4) << "and its symmetric";
  EXPECT_DOUBLE_EQ(pose(1, 5), 0.0);
}

// Against the definition: sample perturbed states, rotate by Eigen's own
// quaternion product, and compare the sample covariance of the position and of
// the rotation vector about the world frame's fixed axes with the published one.
TEST(PoseCovarianceTest, MatchesTheSampledRotationAboutTheFixedAxes) {
  std::mt19937 generator{42};
  std::normal_distribution<double> normal;

  const Eigen::Quaterniond q =
      Eigen::Quaterniond{Eigen::AngleAxisd{0.9, Eigen::Vector3d{1.0, -2.0, 0.5}.normalized()}};
  StateVector state = StateVector::Zero();
  state(kQw) = q.w();
  state(kQw + 1) = q.x();
  state(kQw + 2) = q.y();
  state(kQw + 3) = q.z();

  // A correlated covariance of the position and the quaternion.
  Eigen::Matrix<double, 7, 7> a;
  for (int i = 0; i < 49; ++i) a(i) = normal(generator);
  Eigen::Matrix<double, 7, 7> block = a * a.transpose();
  block.topLeftCorner<3, 3>() *= 1e-2;
  block.topRightCorner<3, 4>() *= 1e-4;
  block.bottomLeftCorner<4, 3>() *= 1e-4;
  block.bottomRightCorner<4, 4>() *= 1e-6;
  StateMatrix covariance = StateMatrix::Identity();
  covariance.topLeftCorner<7, 7>() = block;

  const PoseMatrix expected = Untyped(typed::pose_covariance(state, covariance));

  const Eigen::Matrix<double, 7, 7> l = block.llt().matrixL();
  constexpr int kSamples = 200000;
  PoseMatrix sampled = PoseMatrix::Zero();
  for (int n = 0; n < kSamples; ++n) {
    Eigen::Matrix<double, 7, 1> xi;
    for (int i = 0; i < 7; ++i) xi(i) = normal(generator);
    const Eigen::Matrix<double, 7, 1> d = l * xi;
    Eigen::Quaterniond perturbed{q.w() + d(3), q.x() + d(4), q.y() + d(5), q.z() + d(6)};
    perturbed.normalize();
    Eigen::Quaterniond delta = perturbed * q.conjugate();
    if (delta.w() < 0.0) delta.coeffs() *= -1.0;
    const Eigen::AngleAxisd rotation{delta};
    Eigen::Matrix<double, 6, 1> pose;
    pose << d.head<3>(), rotation.angle() * rotation.axis();
    sampled += pose * pose.transpose() / kSamples;
  }

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      EXPECT_NEAR(sampled(i, j), expected(i, j),
                  0.02 * std::sqrt(expected(i, i) * expected(j, j)))
          << "pose covariance element " << i << ", " << j;
    }
  }
}

}  // namespace
