# Contact force at the tool, estimated from servo load.
#
# There is no force/torque sensor on an SO-101. This message carries an
# ESTIMATE derived from STS3215 present-load readings through a fitted static
# model and the Jacobian transpose (see griff.sensing). Consumers must treat it
# as such: `condition_number` reports how well the arm's current configuration
# can observe force at all, and near a singularity a small reading means the
# estimate is blind, not that nothing is being touched.

std_msgs/Header header

# Estimated contact force in the frame named by header.frame_id, newtons.
geometry_msgs/Vector3 force

# Magnitude, provided so consumers do not each recompute it slightly differently.
float64 magnitude

# Condition number of J*J^T at the current configuration. Large means at least
# one Cartesian direction is poorly observed.
float64 condition_number

# Residual joint torques after the static model was subtracted, N.m. Published
# for diagnosis: a residual that is large while the force is small means the
# calibration no longer describes the arm.
float64[] residual_torque
