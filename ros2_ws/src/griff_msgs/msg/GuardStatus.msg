# What the force-limited admittance controller did to the policy's command.
#
# Published every control cycle so that "the controller bounded the contact" is
# an observable claim rather than an assumption. If `governed` is never true
# during a run that overloaded the workpiece, the guard was not the thing that
# failed -- the force estimate was.

std_msgs/Header header

# True on cycles where the reference governor clipped or reversed the commanded
# advance because the measured force was at or above the limit.
bool governed

# Force limit currently in force, newtons.
float64 force_limit

# Measured (estimated) contact force magnitude this cycle, newtons.
float64 measured_force

# Online estimate of the environment stiffness along the contact normal, N/m.
# This is what the advance cap is computed from; a wildly wrong value here is
# the first thing to look at when the transient overshoots.
float64 environment_stiffness

# Cartesian distance between what the policy asked for and what was commanded,
# metres. Zero in free space.
float64 correction

# Accumulated count of governed cycles since the controller was activated.
uint64 governed_cycles
