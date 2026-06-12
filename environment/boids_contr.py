# environment/boids_controller.py
import numpy as np
from utils import config

def is_visible(drone_pos, forward, target_pos, detection_range, fov_degrees):
    """True if target_pos is within detection_range AND inside the camera cone
    around `forward`. Pass a maintained heading for `forward`, not the raw
    velocity: a hovering drone's velocity direction is numerical noise."""
    to_target = target_pos - drone_pos
    distance = np.linalg.norm(to_target)

    if distance > detection_range or distance < 1e-8:
        return False

    forward_dir = forward / (np.linalg.norm(forward) + 1e-8)
    to_target_dir = to_target / distance
    
    # Dot product gives the cosine of the angle between them
    dot_product = np.clip(np.dot(forward_dir, to_target_dir), -1, 1)
    fov_threshold = np.cos(np.radians(fov_degrees) / 2)
    
    return dot_product >= fov_threshold

def compute_boids_velocity(positions, velocities, headings, fpv_pos, fpv_vel, target_pos, actions):
    """
    Computes the desired 3D velocity for each drone based on Boids rules + RL weights.

    Args:
        positions: (N, 3) array of drone positions
        velocities: (N, 3) array of drone velocities
        headings: (N, 3) array of unit camera-forward directions (last motion heading)
        fpv_pos: (3,) array of FPV position
        fpv_vel: (3,) array of FPV velocity
        target_pos: (3,) array of target position
        actions: (N, 7) array of RL weights
                 [w_pred, w_react, w_pres, w_coh, w_sep, w_guard, w_orbit]
                 First six are in [0, 1]; w_orbit is SIGNED in [-1, 1] so each
                 drone picks its own orbit direction (lets dispersal emerge).

    Returns:
        desired_velocities: (N, 3) array of desired velocity vectors for each drone
    """
    N = positions.shape[0]
    desired_velocities = np.zeros((N, 3))
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8) if norm > 1e-8 else np.zeros(3)
    # Define perception radii for the Boids rules
    COHESION_RADIUS = 15.0
    SEPARATION_RADIUS = 1.5 # e.g., 1.5m

    for i in range(N):
        pos_i = positions[i]
        vel_i = velocities[i]
        fpv_visible = is_visible(pos_i, headings[i], fpv_pos, config.DETECTION_RANGE, config.CAMERA_FOV_DEGREES)
        
        if fpv_visible:
            # 1. Predictive Intercept Vector:Lead the target
            estimated_time = min(np.linalg.norm(fpv_pos - pos_i) / config.MAX_VEL_CMD, 2)
            predicted_pos = fpv_pos + (fpv_vel * estimated_time)
            predictive_vec = normalize(predicted_pos - pos_i)
            # 2. Reactive Intercept Vector: Direction toward the FPV
            reactive_vec = (fpv_pos - pos_i)
            reactive_vec = normalize(reactive_vec)
        else:
            predictive_vec = np.zeros(3)
            reactive_vec = np.zeros(3)
        # 3. Preserve Vector: Direction AWAY from the FPV if it's too close
        if fpv_visible and np.linalg.norm(fpv_pos - pos_i) < config.FPV_DODGE_DIST:
            preserve_vec = normalize(pos_i - fpv_pos)
        else:
            preserve_vec = np.zeros(3)
        
        # 4. Cohesion Vector: Direction toward the center of mass of NEARBY drones
        neighbors = [positions[j] for j in range(N)
                     if j != i and np.linalg.norm(positions[j] - pos_i) < COHESION_RADIUS]
        if neighbors:
            cohesion_vec = normalize(np.mean(neighbors, axis=0) - pos_i)
        else:
            cohesion_vec = np.zeros(3)
        # 5. Separation Vector: Direction AWAY from very close drones to prevent collisions
        separation_vec = np.zeros(3)
        for j in range(N):
            if i != j and np.linalg.norm(pos_i - positions[j]) < SEPARATION_RADIUS:
                repulsion_vec = normalize(pos_i - positions[j])
                separation_vec += repulsion_vec
        separation_vec = normalize(separation_vec)

        # 6. Guardian Vector: spring-like tether to a patrol BAND around the target.
        # A point attractor made the whole swarm collapse onto the asset; the band
        # attracts from outside, repels from inside, and is zero on the shell, so
        # the equilibrium is a spread-out shell at PATROL_RADIUS instead of a ball.
        radial = target_pos - pos_i
        dist_to_target = np.linalg.norm(radial)
        guardian_vec = normalize(radial) * np.tanh(
            (dist_to_target - config.PATROL_RADIUS) / config.PATROL_WIDTH)

        # 7. Orbit Vector: horizontal tangential sweep around the target. Since the
        # camera heading follows the velocity, orbiting doubles as scanning.
        # w_orbit is signed: the policy chooses CW vs CCW per drone, so a clump
        # can split and spread along the shell without hardcoded roles.
        orbit_vec = normalize(np.cross(np.array([0.0, 0.0, 1.0]), pos_i - target_pos))

        # Combine vectors using the RL action weights
        w_pred, w_react, w_pres, w_coh, w_sep, w_guard, w_orbit = actions[i]

        #Sum Weighted Vectors
        combined_vector = (w_pred*predictive_vec) + (w_react * reactive_vec) + (w_pres * preserve_vec) + (w_coh * cohesion_vec) + (w_sep * separation_vec) + (w_guard * guardian_vec) + (w_orbit * orbit_vec)
        
        # Variable speed: CAP the combined magnitude instead of normalizing it.
        # |combined| >= 1 flies at MAX_VEL_CMD; smaller magnitudes command
        # proportionally slower flight, down to a hover when the rules cancel.
        # (Full-speed normalization made parking on the shell, blocking, and
        # stretching a clump apart dynamically impossible.)
        mag = np.linalg.norm(combined_vector)
        if mag > 1e-8:
            desired_velocities[i] = combined_vector * (
                config.MAX_VEL_CMD * min(mag, 1.0) / mag)
        else:
            desired_velocities[i] = np.zeros(3)

        # Ground-avoidance reflex: never let the controller command the drone into the
        # floor. Below GROUND_SAFE_Z the reflex takes over "climb-first": horizontal
        # chase is scaled down and a strong upward velocity is forced, both growing the
        # lower the drone gets, so downward momentum from a dive is reversed in time.
        if pos_i[2] < config.GROUND_SAFE_Z:
            deficit = min((config.GROUND_SAFE_Z - pos_i[2]) / config.GROUND_SAFE_Z, 1.0)  # 0..1
            desired_velocities[i, 0:2] *= (1.0 - deficit)            # suppress horizontal when low
            desired_velocities[i, 2] = config.GROUND_CLIMB_SPEED * deficit  # force climb
        else:
            # Descent brake: above the reflex zone, never command a sink rate the
            # reflex cannot recover from. Full-speed dives from cruise altitude
            # were punching through the reflex on momentum and killing drones.
            min_vz = -(pos_i[2] - config.GROUND_SAFE_Z) / config.GROUND_BRAKE_TIME
            desired_velocities[i, 2] = max(desired_velocities[i, 2], min_vz)

    return desired_velocities