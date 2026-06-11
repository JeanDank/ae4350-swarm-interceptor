# --- Swarm & FPV Dynamics ---
NUM_DRONES = 12
# CF2X can only HOLD ALTITUDE up to ~2.5 m/s; commanding faster makes it tilt
# so hard it sinks and crashes. Keep all speeds inside that envelope.
# The FPV is FASTER than the swarm: a tail-chase can never close on it, so the
# swarm can only win by positioning itself between the threat and the asset.
SWARM_SPEED = 2.0          # Sustainable interceptor speed (slower than the FPV)
FPV_SPEED = 2.5            # Threat outruns pure pursuit but stays altitude-stable
MAX_VEL_CMD = 2.0          # Max commanded speed for the Boids controller (= SWARM_SPEED)
dt = 0.1
# Control rate of the velocity PID. At 10 Hz the controller sinks during fast
# horizontal flight (only corrects every 0.1s); 48 Hz tracks velocity stably and
# holds altitude. 1 env step == 1/CTRL_FREQ s.
CTRL_FREQ = 48

# --- Distances (Scaled to the ~2.5 m/s speed envelope) ---
BLAST_RADIUS = 3.0         # FPV "breaches" target inside this radius
SWARM_RANGE = 30.0
COLLISION_DIST = 1.5       # Hit window for an interception
FPV_DODGE_DIST = 2.0
GROUND_Z_THRESHOLD = 0.1

# --- Protected target / arena geometry ---
TARGET_ALTITUDE = 3.0      # Protected asset sits OFF the ground (was 0 -> guardian dove into floor)
# FPV spawns OUTSIDE detection range: the swarm gets an unseen approach phase and
# must learn to look out for incoming threats (SCOUT_BONUS pays on first sight).
FPV_SPAWN_MIN_DIST = 24.0
FPV_SPAWN_MAX_DIST = 30.0
FPV_SPAWN_MIN_Z = 2.0
FPV_SPAWN_MAX_Z = 5.0

# --- Swarm spawn (clustered, offset from the asset) ---
# The swarm starts grouped a distance away from the target so the defensive
# formation (transit + spread around the asset) has to emerge each episode.
SWARM_SPAWN_DIST = 8.0         # Cluster centre distance from the target
SWARM_SPAWN_RADIUS = 4.0       # Drones scatter within this radius of the centre
SWARM_SPAWN_MIN_Z = 2.5        # >= GROUND_SAFE_Z so the climb reflex doesn't fight the spawn
SWARM_SPAWN_MAX_Z = 4.5
SWARM_SPAWN_MIN_SPACING = 2.0  # > Boids SEPARATION_RADIUS: no repulsion burst at t=0

# --- Ground-avoidance reflex (applied to swarm desired velocity) ---
GROUND_SAFE_Z = 2.5        # Below this altitude the climb-first reflex kicks in
GROUND_CLIMB_SPEED = 2.5   # Vertical climb is tilt-free, so it may exceed MAX_VEL_CMD

# --- Detection / Sensing ---
DETECTION_RANGE = 20.0     # Scaled to the smaller arena
# Realistic forward-facing camera. Training uses a curriculum that starts at 360
# and narrows to this value (see train_mappo.py); SwarmInterceptEnv(fov_degrees=...)
# overrides it per environment.
CAMERA_FOV_DEGREES = 120.0
HEADING_MIN_SPEED = 0.3    # Below this speed keep the last heading (velocity dir is noise)
FPV_MEMORY_STEPS = 48      # ~1 s: drones remember the last-seen FPV state, decaying in the obs

# --- Wave Management ---
NUM_FPV_WAVES = 2
MIN_WAVE_DELAY_STEPS = 20
MAX_WAVE_DELAY_STEPS = 50
# Each approach from 24-30 m at 2.5 m/s needs ~10-12 s (~500-580 steps); budget
# covers two waves plus the inter-wave delay.
MAX_EPISODE_STEPS = 1200

# --- Engagement Thresholds ---
RAMMING_THRESHOLD = 0.5    # Min RELATIVE closing speed for a kill (blocking in the FPV's path counts)

# --- Rewards and Penalties ---
# Ordering that drives behaviour: protecting the target is worth ANY sacrifice
# (team success dominates everything), and dying is never cheaper than fighting
# on (|SAC_PEN| >= |FAIL_PEN| * NUM_FPV_WAVES closes the suicide loophole).
TIME_PEN = -0.02           # Small: must NOT dominate APPROACH_BONUS, or the agent learns to suicide
SAC_PEN = -400.0           # Death (ground / passive collision). Was -50: cheaper than living
                           # through breaches, so drones learned to climb-and-dive suicide.
FAIL_PEN = -200.0          # Target breach, paid by EVERY living drone
SUCCESS_REWARD = 500.0     # Killer's personal bonus on top of the team share
TEAM_SUCCESS_REWARD = 300.0  # Every living drone shares each interception (no kill lottery)
EARLY_BONUS = 500.0
APPROACH_BONUS = 1.0       # Dense reward for closing on the FPV; the bridge to the sparse +500
SCOUT_BONUS = 50.0
SURVIVOR_REWARD = 100.0    # Episode end, ONLY if the target was never breached

# --- Patrol / formation shaping (paid only while a drone has NO fresh FPV track) ---
PATROL_RADIUS = 10.0       # Guardian rule's equilibrium shell around the target
PATROL_WIDTH = 4.0         # Band softness (tanh scale of the guardian + bonus half-width)
PATROL_BONUS = 0.02        # Per-step bonus for holding the patrol band
SPACING_BONUS = 0.02       # Per-step bonus for distance to the nearest teammate
SPACING_CAP = 6.0          # Spacing bonus saturates here (no reward for scattering further)
NEIGHBOR_RADIUS = 15.0     # Teammates inside this range appear in the neighbour obs

# --- Physical Properties ---
M_FPV = 2.0
M_SWARM = 0.25
