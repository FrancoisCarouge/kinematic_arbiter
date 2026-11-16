#pragma once

#include <sensor_msgs/msg/imu.hpp>
#include "kinematic_arbiter/ros2/sensor_handler.hpp"
#include "kinematic_arbiter/sensors/imu_sensor_model.hpp"
#include "kinematic_arbiter/ros2/ros2_utils.hpp"
namespace kinematic_arbiter {
namespace ros2 {

/**
 * @brief IMU sensor handler
 */
class ImuSensorHandler : public SensorHandler<sensor_msgs::msg::Imu> {
public:
  using Base = SensorHandler<sensor_msgs::msg::Imu>;
  using ModelType = kinematic_arbiter::sensors::ImuSensorModel;
  using MeasurementIndex = ModelType::MeasurementIndex;

  ImuSensorHandler(
      rclcpp::Node* node,
      std::shared_ptr<Filter> filter,
      std::shared_ptr<tf2_ros::Buffer> tf_buffer,
      std::shared_ptr<utils::TimeManager> time_manager,
      const std::string& sensor_name,
      const std::string& topic,
      const std::string& sensor_frame_id,
      const std::string& reference_frame_id,
      const std::string& body_frame_id)
    : Base(node, filter, tf_buffer, time_manager, sensor_name, topic,
          sensor_frame_id, reference_frame_id, body_frame_id,
          std::make_shared<ModelType>(),
          SensorType::Imu) {}

protected:

  bool msgToVector(const sensor_msgs::msg::Imu& msg, DynamicVector& vector) override {
    vector.resize(ModelType::kMeasurementDimension);

    // Angular velocity (gyroscope measurements)
    vector(MeasurementIndex::GX) = msg.angular_velocity.x;
    vector(MeasurementIndex::GY) = msg.angular_velocity.y;
    vector(MeasurementIndex::GZ) = msg.angular_velocity.z;

    // Linear acceleration (accelerometer measurements)
    vector(MeasurementIndex::AX) = msg.linear_acceleration.x;
    vector(MeasurementIndex::AY) = msg.linear_acceleration.y;
    vector(MeasurementIndex::AZ) = msg.linear_acceleration.z;

    return true;
  }

  sensor_msgs::msg::Imu vectorToMsg(
      const DynamicVector& vector,
      const std_msgs::msg::Header& header) override {

    sensor_msgs::msg::Imu msg;
    msg.header = header;

    // Angular velocity
    msg.angular_velocity.x = vector(MeasurementIndex::GX);
    msg.angular_velocity.y = vector(MeasurementIndex::GY);
    msg.angular_velocity.z = vector(MeasurementIndex::GZ);

    // Linear acceleration
    msg.linear_acceleration.x = vector(MeasurementIndex::AX);
    msg.linear_acceleration.y = vector(MeasurementIndex::AY);
    msg.linear_acceleration.z = vector(MeasurementIndex::AZ);

    // Orientation (identity quaternion as we don't produce this)
    msg.orientation.w = 1.0;
    msg.orientation.x = 0.0;
    msg.orientation.y = 0.0;
    msg.orientation.z = 0.0;

    // Set orientation covariance to high uncertainty
    for (int i = 0; i < 9; ++i) {
      msg.orientation_covariance[i] = (i % 4 == 0) ? 99999.0 : 0.0;
    }

    return msg;
  }

  sensor_msgs::msg::Imu applyBound(
      const sensor_msgs::msg::Imu& base_msg,
      const DynamicCovariance& covariance,
      bool positive) override {

    auto bounded_msg = base_msg;
    double sign = positive ? 1.0 : -1.0;

    // Angular velocity bounds using measurement indices
    bounded_msg.angular_velocity.x = SigmaBounded<typed::AngularSpeed>(
        bounded_msg.angular_velocity.x, covariance(MeasurementIndex::GX, MeasurementIndex::GX), sign);
    bounded_msg.angular_velocity.y = SigmaBounded<typed::AngularSpeed>(
        bounded_msg.angular_velocity.y, covariance(MeasurementIndex::GY, MeasurementIndex::GY), sign);
    bounded_msg.angular_velocity.z = SigmaBounded<typed::AngularSpeed>(
        bounded_msg.angular_velocity.z, covariance(MeasurementIndex::GZ, MeasurementIndex::GZ), sign);

    // Linear acceleration bounds using measurement indices
    bounded_msg.linear_acceleration.x = SigmaBounded<typed::Acceleration>(
        bounded_msg.linear_acceleration.x, covariance(MeasurementIndex::AX, MeasurementIndex::AX), sign);
    bounded_msg.linear_acceleration.y = SigmaBounded<typed::Acceleration>(
        bounded_msg.linear_acceleration.y, covariance(MeasurementIndex::AY, MeasurementIndex::AY), sign);
    bounded_msg.linear_acceleration.z = SigmaBounded<typed::Acceleration>(
        bounded_msg.linear_acceleration.z, covariance(MeasurementIndex::AZ, MeasurementIndex::AZ), sign);

    return bounded_msg;
  }
};

} // namespace ros2
} // namespace kinematic_arbiter
