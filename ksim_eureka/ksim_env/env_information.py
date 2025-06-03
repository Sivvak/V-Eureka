def get_available_observations() -> str:
    """Return a string describing available observations in K-Sim."""
    return """
- joint_position_observation: Joint positions
- joint_velocity_observation: Joint velocities  
- actuator_force_observation: Actuator forces
- center_of_mass_inertia_observation: Center of mass inertia
- center_of_mass_velocity_observation: Center of mass velocity
- base_position_observation: Base position in world frame
- base_orientation_observation: Base orientation quaternion
- base_linear_velocity_observation: Base linear velocity
- base_angular_velocity_observation: Base angular velocity
- base_linear_acceleration_observation: Base linear acceleration
- base_angular_acceleration_observation: Base angular acceleration
- projected_gravity_observation: Gravity vector in robot frame
- sensor_observation_imu_acc: IMU acceleration
- sensor_observation_imu_gyro: IMU gyroscope
- timestep_observation: Current timestep
- feet_contact_observation: Foot contact information
- feet_position_observation: Foot positions
- feet_orientation_observation: Foot orientations
"""
