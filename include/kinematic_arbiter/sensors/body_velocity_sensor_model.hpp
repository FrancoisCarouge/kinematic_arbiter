#pragma once

#include "kinematic_arbiter/core/measurement_model_interface.hpp"
#include "kinematic_arbiter/core/state_index.hpp"
#include "kinematic_arbiter/core/sensor_types.hpp"
#include "kinematic_arbiter/core/typed_state.hpp"

namespace kinematic_arbiter {
namespace sensors {

/**
 * @brief Body velocity measurement model (linear + angular velocity)
 *
 * Models a sensor that measures linear and angular velocities in the body frame.
 * Measurement vector is [vx, vy, vz, wx, wy, wz]' where
 * [vx, vy, vz] represents linear velocity and [wx, wy, wz] represents angular velocity.
 */
class BodyVelocitySensorModel : public core::MeasurementModelInterface {
public:
  // Type definitions for clarity
  using Base = core::MeasurementModelInterface;
  static constexpr int kMeasurementDimension = core::MeasurementDimension<core::SensorType::BodyVelocity>::value;
  using StateVector = typename Base::StateVector;
  using Vector = Eigen::Matrix<double, kMeasurementDimension, 1>;
  using Jacobian = Eigen::Matrix<double, kMeasurementDimension, core::StateIndex::kFullStateSize>;
  using Covariance = Eigen::Matrix<double, kMeasurementDimension, kMeasurementDimension>;
  using StateFlags = typename Base::StateFlags;
  /// The linear velocity in m/s, the angular velocity in rad/s.
  using Elements = decltype(std::tuple_cat(typed::Repeat<typed::Speed, 3>{}, typed::Repeat<typed::AngularSpeed, 3>{}));
  /// The linear velocity depends on the linear and, through the lever arm, the
  /// angular velocities; the angular velocity on the angular velocity.
  static constexpr fcarouge::sparse::pattern kJacobianPattern{
      fcarouge::sparse::shape<6, core::StateIndex::kFullStateSize>,
      [](std::size_t row, std::size_t column) {
        return column >= (row < 3U ? core::StateIndex::LinearVelocity::X : core::StateIndex::AngularVelocity::X) &&
               column <= core::StateIndex::AngularVelocity::Z;
      }};

  /**
   * @brief Indices for accessing body velocity measurement components
   */
  struct MeasurementIndex {
    // Linear velocity indices
    static constexpr int VX = 0;
    static constexpr int VY = 1;
    static constexpr int VZ = 2;

    // Angular velocity indices
    static constexpr int WX = 3;
    static constexpr int WY = 4;
    static constexpr int WZ = 5;
  };

  /**
   * @brief Constructor
   *
   * @param sensor_pose_in_body_frame Transform from body to sensor frame
   * @param params Validation parameters
   */
  explicit BodyVelocitySensorModel(
      const Eigen::Isometry3d& sensor_pose_in_body_frame = Eigen::Isometry3d::Identity(),
      const ValidationParams& params = ValidationParams())
    : Base(core::SensorType::BodyVelocity, sensor_pose_in_body_frame, params, Covariance::Identity()) {}

    /**
     * @brief Reset the measurement model to initial state
     */
    void reset() override {
      measurement_covariance_ = Covariance::Identity();
    }

  /**
   * @brief Predict measurement from state
   *
   * @param state Current state estimate
   * @return Expected measurement [vx, vy, vz, wx, wy, wz]'
   */
  DynamicVector PredictMeasurement(const StateVector& state) const override {
    using typed::AngularSpeed;
    using typed::Speed;

    Vector predicted_measurement = Vector::Zero();

    // Extract linear velocity from state
    const auto body_lin_vel = typed::typed<Speed>(Eigen::Vector3d{state.segment<3>(core::StateIndex::LinearVelocity::X)});

    // Extract angular velocity from state
    const auto body_ang_vel = typed::typed<AngularSpeed>(Eigen::Vector3d{state.segment<3>(core::StateIndex::AngularVelocity::X)});

    // Extract the sensor-to-body transform components
    const auto trans_b_s = typed::typed<typed::Length>(Eigen::Vector3d{sensor_pose_in_body_frame_.translation()});
    const Eigen::Matrix3d rot_s_b = sensor_pose_in_body_frame_.rotation().transpose();

    // Calculate linear velocity at the sensor location in body frame
    // v_sensor = v_body + ω × r, where r is the position of the sensor in the body frame,
    // the cross product by the skew-symmetric matrix of ω
    const typed::Vector<Speed> body_sensor_lin_vel{
        body_lin_vel + typed::skew<Speed, typed::Length>(body_ang_vel) * trans_b_s};

    // Transform velocities to sensor frame
    const typed::Vector<Speed> sensor_lin_vel{
        typed::typed<typed::Rotation<Speed>>(rot_s_b, mp_units::one) * body_sensor_lin_vel};
    const typed::Vector<AngularSpeed> sensor_ang_vel{
        typed::typed<typed::Rotation<AngularSpeed>>(rot_s_b, mp_units::one) * body_ang_vel};

    // Fill measurement vector
    typed::write<MeasurementIndex::VX>(predicted_measurement, sensor_lin_vel);
    typed::write<MeasurementIndex::WX>(predicted_measurement, sensor_ang_vel);

    return predicted_measurement;
  }

  /**
   * @brief Compute measurement Jacobian
   *
   * @param state Current state estimate
   * @return Jacobian of measurement with respect to state
   */
  DynamicJacobian GetMeasurementJacobian(const StateVector&) const override {
    using typed::AngularSpeed;
    using typed::Speed;

    // The linear velocity in m/s, the angular velocity in rad/s: each partial
    // derivative in its measurement's unit per its state's unit.
    typed::Jacobian<Elements, kJacobianPattern> jacobian;

    // Extract transform parameters
    const auto sensor_offset = typed::typed<typed::Length>(Eigen::Vector3d{sensor_pose_in_body_frame_.translation()});
    const Eigen::Matrix3d body_to_sensor_rotation = sensor_pose_in_body_frame_.rotation().transpose();
    const auto body_to_sensor_speed = typed::typed<typed::Rotation<Speed>>(body_to_sensor_rotation, mp_units::one);

    // ===== LINEAR VELOCITY JACOBIAN =====

    // Linear velocity w.r.t. linear velocity - simple rotation from body to sensor frame
    typed::place<MeasurementIndex::VX, core::StateIndex::LinearVelocity::X>(jacobian, body_to_sensor_speed);

    // Linear velocity w.r.t. angular velocity - lever arm effect
    // The lever arm effect can be represented by partial derivatives of the cross product
    // These derivatives have a standard form for cross products

    // Following standard notation where [r×] is the skew-symmetric matrix of vector r
    // For a sensor at position r, the derivatives of ω×r with respect to each ω component
    // correspond to the columns of -[r×], in metres: the lever arm effect of ω_x, ω_y, ω_z
    const typed::Partial<Speed, AngularSpeed> lever_arm_effect{
        -typed::skew<Speed, AngularSpeed>(sensor_offset)};

    // Apply body-to-sensor rotation and fill Jacobian
    typed::place<MeasurementIndex::VX, core::StateIndex::AngularVelocity::X>(
        jacobian, typed::Partial<Speed, AngularSpeed>{body_to_sensor_speed * lever_arm_effect});

    // ===== ANGULAR VELOCITY JACOBIAN =====

    // Angular velocity w.r.t. angular velocity - simple rotation from body to sensor frame
    typed::place<MeasurementIndex::WX, core::StateIndex::AngularVelocity::X>(
        jacobian, typed::typed<typed::Rotation<AngularSpeed>>(body_to_sensor_rotation, mp_units::one));

    return typed::untyped<decltype(jacobian), DynamicJacobian>(jacobian);
  }

    /**
   * @brief Project the state covariance into the measurement space: C_k P_k C_k^T
   */
  DynamicCovariance ProjectStateCovariance(
      const DynamicJacobian& jacobian, const StateCovariance& state_covariance) const override {
    return typed::project<Elements, kJacobianPattern, DynamicCovariance>(jacobian, state_covariance);
  }

  /**
   * @brief Get states that this sensor can directly initialize
   *
   * Body velocity sensors can initialize linear and angular velocity states.
   *
   * @return Flags for initializable states
   */
  StateFlags GetInitializableStates() const override {
    StateFlags flags = StateFlags::Zero();

    // Body velocity sensor can initialize all velocity states
    flags[core::StateIndex::LinearVelocity::X] = true;
    flags[core::StateIndex::LinearVelocity::Y] = true;
    flags[core::StateIndex::LinearVelocity::Z] = true;
    flags[core::StateIndex::AngularVelocity::X] = true;
    flags[core::StateIndex::AngularVelocity::Y] = true;
    flags[core::StateIndex::AngularVelocity::Z] = true;

    return flags;
  }

  /**
   * @brief Initialize state from body velocity measurement
   *
   * Initializes linear and angular velocity states based on the measurement,
   * properly accounting for lever arm effects between sensor and body frame.
   *
   * @param measurement Velocity measurement [vx, vy, vz, wx, wy, wz]
   * @param valid_states Flags indicating which states are valid for initialization
   * @param state State vector to update
   * @param covariance State covariance to update
   * @return Flags indicating which states were initialized
   */
  StateFlags InitializeState(
      const DynamicVector& measurement,
      const StateFlags&,
      StateVector& state,
      StateCovariance& covariance) const override {
    ValidateMeasurementSize(measurement);

    StateFlags initialized_states = StateFlags::Zero();

    // Extract measurement components in sensor frame
    const Eigen::Vector3d sensor_lin_vel = measurement.segment<3>(0);
    const Eigen::Vector3d sensor_ang_vel = measurement.segment<3>(3);

      // Extract the sensor-to-body transform components
      const Eigen::Vector3d trans_b_s = sensor_pose_in_body_frame_.translation();
      const Eigen::Matrix3d rot_b_s = sensor_pose_in_body_frame_.rotation();

      // Calculate the skew-symmetric matrix for the cross product: ω × r
      auto skew_matrix = [](const Eigen::Vector3d& v) -> Eigen::Matrix3d {
        Eigen::Matrix3d skew;
        skew << 0, -v(2), v(1),
                v(2), 0, -v(0),
                -v(1), v(0), 0;
        return skew;
      };

      // First, compute body angular velocity
      const Eigen::Vector3d body_ang_vel = rot_b_s * sensor_ang_vel;

      // Convert sensor linear velocity to body frame
      const Eigen::Vector3d sensor_lin_vel_in_body = rot_b_s * sensor_lin_vel;

      // Account for the lever arm effect
      const Eigen::Vector3d lever_arm_effect = skew_matrix(body_ang_vel) * trans_b_s;

      // Compute body linear velocity
      const Eigen::Vector3d body_lin_vel = sensor_lin_vel_in_body - lever_arm_effect;

      // Initialize state vector
      state.segment<3>(core::StateIndex::LinearVelocity::Begin()) = body_lin_vel;
      state.segment<3>(core::StateIndex::AngularVelocity::Begin()) = body_ang_vel;

      // Set all velocity states as initialized
      initialized_states[core::StateIndex::LinearVelocity::X] = true;
      initialized_states[core::StateIndex::LinearVelocity::Y] = true;
      initialized_states[core::StateIndex::LinearVelocity::Z] = true;
      initialized_states[core::StateIndex::AngularVelocity::X] = true;
      initialized_states[core::StateIndex::AngularVelocity::Y] = true;
      initialized_states[core::StateIndex::AngularVelocity::Z] = true;

      // Transform covariance from sensor to body frame
      // For angular velocity (straightforward rotation)
      Eigen::Matrix3d ang_vel_cov = rot_b_s *
          measurement_covariance_.block<3, 3>(3, 3) *
          rot_b_s.transpose();

      // For linear velocity (approximate - ignoring cross-terms with angular velocity)
      Eigen::Matrix3d lin_vel_cov = rot_b_s *
          measurement_covariance_.block<3, 3>(0, 0) *
          rot_b_s.transpose();

      // Set covariance blocks in state covariance
      covariance.block<3, 3>(
          core::StateIndex::LinearVelocity::Begin(),
          core::StateIndex::LinearVelocity::Begin()) = lin_vel_cov;

      covariance.block<3, 3>(
          core::StateIndex::AngularVelocity::Begin(),
          core::StateIndex::AngularVelocity::Begin()) = ang_vel_cov;


    return initialized_states;
  }


};

} // namespace sensors
} // namespace kinematic_arbiter
