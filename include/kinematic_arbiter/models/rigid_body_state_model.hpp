#pragma once

#include "kinematic_arbiter/core/state_model_interface.hpp"
#include "kinematic_arbiter/core/typed_state.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>

namespace kinematic_arbiter {
namespace models {
namespace {
  const double kLambda = 46.05;
}


/**
 * @brief State model for 3D rigid body with quaternion orientation
 *
 * Implements a state transition model for a rigid body in 3D space with:
 * - 3D position
 * - Quaternion orientation
 * - 3D linear velocity
 * - 3D angular velocity
 * - 3D linear acceleration
 * - 3D angular acceleration
 *
 * The model uses quaternion kinematics for rotation representation to avoid gimbal lock.
 */
class RigidBodyStateModel : public core::StateModelInterface {
public:
  // Use convenient matrix/vector aliases for 3D/4D operations
  using Matrix3d = Eigen::Matrix<double, 3, 3>;
  using Vector3d = Eigen::Vector3d;
  using Vector4d = Eigen::Vector4d;
  using Matrix4d = Eigen::Matrix<double, 4, 4>;

  // Type alias for cleaner access to state indices
  using SIdx = core::StateIndex;

  /**
   * @brief Construct a new RigidBodyStateModel
   * @param params State model parameters
   */
  explicit RigidBodyStateModel(const Params& params = Params())
    : core::StateModelInterface(params) {}

  /**
   * @brief Predict state forward in time: x̂_k^- = f(x̂_{k-1}^+, u_k)
   *
   * @param current_state Current state estimate x̂_{k-1}^+
   * @param time_step Time step in seconds
   * @return Predicted next state x̂_k^-
   */
    StateVector PredictState(const StateVector& current_state, double time_step) const override {
    StateVector new_states = current_state;

    if (fabs(time_step) <= std::numeric_limits<double>::epsilon()) {
      return new_states;
    }

    // Extract quaternion and normalize
    Eigen::Quaterniond orientation(
      current_state[SIdx::Quaternion::W],
      current_state[SIdx::Quaternion::X],
      current_state[SIdx::Quaternion::Y],
      current_state[SIdx::Quaternion::Z]
    );
    orientation.normalize();

    // Get rotation matrix
    Matrix3d rotation_matrix_b_to_w = orientation.toRotationMatrix();

    // Extract velocity and acceleration components
    const Vector3d linear_velocity =
      current_state.segment<3>(SIdx::LinearVelocity::Begin());

    const Vector3d angular_velocity =
      current_state.segment<3>(SIdx::AngularVelocity::Begin());

    const Vector3d linear_acceleration =
      current_state.segment<3>(SIdx::LinearAcceleration::Begin());

    const Vector3d angular_acceleration =
      current_state.segment<3>(SIdx::AngularAcceleration::Begin());

    // Linear XYZ Position Prediction Model
    new_states.segment<3>(SIdx::Position::Begin()) +=
      rotation_matrix_b_to_w * ((linear_velocity + time_step * 0.5 * linear_acceleration) * time_step);

    // Angular Position (Quaternion) Prediction Model
    // Use quaternion kinematics with angular acceleration: q(t+dt) = exp(0.5*ω*dt + 0.25*ω̇*dt²) * q(t)
    Eigen::Quaterniond delta_q;

    // Calculate combined angular motion vector v = 0.5*ω*dt + 0.25*ω̇*dt²
    // This is the same vector used in the quaternion update: q(t+dt) = exp(v) * q(t)
    double t = time_step;
    Vector3d combined_angular_motion = angular_velocity * t + 0.5 * angular_acceleration * t * t;
    double motion_norm = combined_angular_motion.norm();

    if (motion_norm > std::numeric_limits<double>::min()) {
      Vector3d axis = combined_angular_motion / motion_norm;
      double angle = motion_norm;
      delta_q = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    } else {
      delta_q = Eigen::Quaterniond::Identity();
    }

    Eigen::Quaterniond new_quaternion = delta_q * orientation;
    new_quaternion.normalize();

    new_states[SIdx::Quaternion::W] = new_quaternion.w();
    new_states[SIdx::Quaternion::X] = new_quaternion.x();
    new_states[SIdx::Quaternion::Y] = new_quaternion.y();
    new_states[SIdx::Quaternion::Z] = new_quaternion.z();

    // Velocity prediction (integrate acceleration)
    new_states.segment<3>(SIdx::LinearVelocity::Begin()) += time_step * linear_acceleration;
    new_states.segment<3>(SIdx::AngularVelocity::Begin()) += time_step * angular_acceleration;
    // Acceleration is modelled as exponential decay: a(t+dt) = a(t) * exp(-lambda * dt)
    // With lambda = 46.05, acceleration decays to near zero within 0.1s
    new_states.segment<3>(SIdx::LinearAcceleration::Begin()) =
        linear_acceleration * std::exp(-kLambda * time_step);
    new_states.segment<3>(SIdx::AngularAcceleration::Begin()) =
        angular_acceleration * std::exp(-kLambda * time_step);

    return new_states;
  }

  /**
   * @brief Get the state transition matrix: A_k
   *
   * The matrix A_k linearizes the state transition function:
   * A_k = ∂f/∂x evaluated at x̂_{k-1}^+ and u_k
   *
   * @param current_state Current state estimate x̂_{k-1}^+
   * @param time_step Time step in seconds
   * @return State transition matrix A_k
   */
  StateMatrix GetTransitionMatrix(const StateVector& current_state, double time_step) const override {
    if (fabs(time_step) <= std::numeric_limits<double>::epsilon()) {
      return StateMatrix::Identity();
    }

    using typed::Acceleration;
    using typed::AngularAcceleration;
    using typed::AngularSpeed;
    using typed::Length;
    using typed::Partial;
    using typed::Ratio;
    using typed::Speed;

    // Each element in its row's state unit per its column's.
    typed::Transition<kTransitionPattern> jacobian;
    const auto t = time_step * typed::second;
    const Ratio one{1.0 * mp_units::one};

    // Extract quaternion and normalize
    Eigen::Quaterniond orientation(
      current_state[SIdx::Quaternion::W],
      current_state[SIdx::Quaternion::X],
      current_state[SIdx::Quaternion::Y],
      current_state[SIdx::Quaternion::Z]
    );
    orientation.normalize();

    // Get rotation matrix (body to world frame), as PredictState
    Matrix3d rotation_matrix_b_to_w = orientation.toRotationMatrix();

    // Position block - identity for position elements
    typed::place_diagonal<SIdx::Position::Begin(), SIdx::Position::Begin(), 3>(jacobian, one);

    // Position wrt linear velocity, in seconds: the body velocity rotated to the world
    typed::place<SIdx::Position::Begin(), SIdx::LinearVelocity::Begin()>(
      jacobian, typed::typed<Partial<Length, Speed>>(rotation_matrix_b_to_w, t));

    // Position wrt linear acceleration, in squared seconds
    typed::place<SIdx::Position::Begin(), SIdx::LinearAcceleration::Begin()>(
      jacobian, typed::typed<Partial<Length, Acceleration>>(rotation_matrix_b_to_w, t * t * 0.5));

    // Remove coupling between linear and angular position in the jacobian by
    // leaving it at zero, structurally. Derivation of partial of position with
    // respect to the rotation in quaternion form can be found here:
    // http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf

    // Extract angular velocity and acceleration components
    const Vector3d angular_velocity =
      current_state.segment<3>(SIdx::AngularVelocity::Begin());

    const Vector3d angular_acceleration =
      current_state.segment<3>(SIdx::AngularAcceleration::Begin());

    // The quaternion update of PredictState: q(t+dt) = dq(v) * q(t), the rotation vector
    // v = w*t + 0.5*wdot*t^2, dq(v) = [cos(|v|/2), v/|v| sin(|v|/2)], q(t) normalized.
    const Vector3d v = angular_velocity * time_step + 0.5 * angular_acceleration * time_step * time_step;
    const double v_norm = v.norm();

    // The derivative of dq wrt v, dimensionless
    Eigen::Matrix<double, 4, 3> ddq_dv;
    ddq_dv.row(0) = -0.25 * v.transpose();
    ddq_dv.bottomRows<3>() = 0.5 * Matrix3d::Identity();
    if (v_norm > 1e-8) {
      const double dq_w = std::cos(0.5 * v_norm);
      const double sinc_half = std::sin(0.5 * v_norm) / v_norm;
      ddq_dv.row(0) = -0.5 * sinc_half * v.transpose();
      ddq_dv.bottomRows<3>() = sinc_half * Matrix3d::Identity() +
          (0.5 * dq_w - sinc_half) / (v_norm * v_norm) * v * v.transpose();
    }

    // The product dq * q as a linear map of dq, in w, x, y, z order
    Matrix4d right;
    right << orientation.w(), -orientation.x(), -orientation.y(), -orientation.z(),
             orientation.x(), orientation.w(), orientation.z(), -orientation.y(),
             orientation.y(), -orientation.z(), orientation.w(), orientation.x(),
             orientation.z(), orientation.y(), -orientation.x(), orientation.w();

    // Quaternion self-propagation, to first order in the rotation: the rotation itself
    // is carried by the angular velocity and acceleration blocks, so that A * x
    // approximates the prediction
    typed::place_diagonal<SIdx::Quaternion::Begin(), SIdx::Quaternion::Begin(), 4>(jacobian, one);

    // Quaternion wrt angular velocity, in seconds, and angular acceleration, in
    // squared seconds, through v
    const Eigen::Matrix<double, 4, 3> dq_dv = right * ddq_dv;
    typed::place<SIdx::Quaternion::Begin(), SIdx::AngularVelocity::Begin()>(
      jacobian, typed::typed<Partial<Ratio, AngularSpeed, 3, 4>>(dq_dv, t));
    typed::place<SIdx::Quaternion::Begin(), SIdx::AngularAcceleration::Begin()>(
      jacobian, typed::typed<Partial<Ratio, AngularAcceleration, 3, 4>>(dq_dv, 0.5 * t * t));

    // Velocity blocks - identity for velocity elements
    typed::place_diagonal<SIdx::LinearVelocity::Begin(), SIdx::LinearVelocity::Begin(), 3>(jacobian, one);

    // Linear velocity wrt linear acceleration
    typed::place_diagonal<SIdx::LinearVelocity::Begin(), SIdx::LinearAcceleration::Begin(), 3>(jacobian, t);

    // Angular velocity blocks - identity for angular velocity elements
    typed::place_diagonal<SIdx::AngularVelocity::Begin(), SIdx::AngularVelocity::Begin(), 3>(jacobian, one);

    // Angular velocity wrt angular acceleration
    typed::place_diagonal<SIdx::AngularVelocity::Begin(), SIdx::AngularAcceleration::Begin(), 3>(jacobian, t);

    // Acceleration is modelled as exponential decay
    const Ratio decay{std::exp(-kLambda * time_step) * mp_units::one};
    typed::place_diagonal<SIdx::AngularAcceleration::Begin(), SIdx::AngularAcceleration::Begin(), 3>(jacobian, decay);
    typed::place_diagonal<SIdx::LinearAcceleration::Begin(), SIdx::LinearAcceleration::Begin(), 3>(jacobian, decay);

    return typed::untyped<decltype(jacobian), StateMatrix>(jacobian);
  }

  /**
   * @brief Propagate the state covariance through the typed transition matrix,
   * skipping its structural zeros: A P Aᵀ + Q
   */
  StateMatrix PropagateCovariance(
      const StateMatrix& transition, const StateMatrix& covariance,
      const StateMatrix& process_noise) const override {
    return typed::propagate<kTransitionPattern, StateMatrix>(transition, covariance, process_noise);
  }

private:
  /// The positions the transition depends on: the position on itself and, through
  /// the body to world rotation, on the linear velocity and acceleration; the
  /// quaternion on itself and on the angular velocity and acceleration; each
  /// velocity on itself and its acceleration; each acceleration on itself.
  static constexpr fcarouge::sparse::pattern kTransitionPattern{
      fcarouge::sparse::shape<core::StateIndex::kFullStateSize, core::StateIndex::kFullStateSize>,
      [](std::size_t row, std::size_t column) {
        using S = core::StateIndex;
        const auto within = [](std::size_t index, std::size_t begin, std::size_t size) {
          return index >= begin && index < begin + size;
        };
        if (row == column) {
          return true;
        }
        if (within(row, S::Position::Begin(), 3)) {
          return within(column, S::LinearVelocity::Begin(), 3) ||
                 within(column, S::LinearAcceleration::Begin(), 3);
        }
        if (within(row, S::Quaternion::Begin(), 4)) {
          return within(column, S::AngularVelocity::Begin(), 3) ||
                 within(column, S::AngularAcceleration::Begin(), 3);
        }
        if (within(row, S::LinearVelocity::Begin(), 6)) {
          return column == row + 6;
        }
        return false;
      }};
};

} // namespace models
} // namespace kinematic_arbiter
