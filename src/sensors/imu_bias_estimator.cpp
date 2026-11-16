#include "kinematic_arbiter/sensors/imu_bias_estimator.hpp"

namespace kinematic_arbiter {
namespace sensors {

void ImuBiasEstimator::EstimateBiases(
    const Eigen::Vector3d& measured_gyro,
    const Eigen::Vector3d& measured_accel,
    const Eigen::Vector3d& predicted_gyro,
    const Eigen::Vector3d& predicted_accel) {


  // Calculate instantaneous bias estimate (measured - predicted)
  using typed::Acceleration;
  using typed::AngularSpeed;
  const typed::Vector<AngularSpeed> current_gyro_bias{
      typed::typed<AngularSpeed>(measured_gyro) - typed::typed<AngularSpeed>(predicted_gyro)};
  const typed::Vector<Acceleration> current_accel_bias{
      typed::typed<Acceleration>(measured_accel) - typed::typed<Acceleration>(predicted_accel)};

  // Update biases using recursive averaging formula
  gyro_bias_ = typed::Vector<AngularSpeed>{gyro_bias_ + (current_gyro_bias - gyro_bias_) / window_size_};
  accel_bias_ = typed::Vector<Acceleration>{accel_bias_ + (current_accel_bias - accel_bias_) / window_size_};
}

} // namespace sensors
} // namespace kinematic_arbiter
