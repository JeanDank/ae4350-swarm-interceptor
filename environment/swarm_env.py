# environment/swarm_env.py
import numpy as np
import pybullet as p
from gymnasium.spaces import Box
from pettingzoo.utils.env import ParallelEnv
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel
from utils import config
from environment.boids_contr import compute_boids_velocity, is_visible


class SwarmInterceptEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "swarm_intercept_v0"}

    def __init__(self, render_mode=None, fov_degrees=None):
        super().__init__()
        self.num_drones = config.NUM_DRONES
        self.max_waves = config.NUM_FPV_WAVES
        self.render_mode = render_mode
        # FOV override for curriculum training. Mutating the config module keeps
        # boids_contr (which reads config.CAMERA_FOV_DEGREES at call time) consistent
        # with this env; envs live in separate Ray worker processes, so this is safe.
        if fov_degrees is not None:
            config.CAMERA_FOV_DEGREES = float(fov_degrees)

        # Initialize simulator with 1 extra drone for the FPV.
        # Drones are driven by VelocityAviary's internal DSL-PID velocity tracking.
        self.sim_env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_drones + 1,
            initial_xyzs=np.vstack([
                np.random.uniform([-10, -10, 2], [10, 10, 5], (self.num_drones, 3)),
                np.array([[12, 0, 3]])  # FPV placeholder; reset() re-spawns it in detection range
            ]),
            pyb_freq=240,
            ctrl_freq=config.CTRL_FREQ,  # 48 Hz: stable velocity tracking that holds altitude
            gui=(render_mode == "human"),
            record=False
        )
        # VelocityAviary's default SPEED_LIMIT (~0.25 m/s) is far below the speeds
        # the config/rewards assume. Raise it so commanded velocities can actually
        # reach FPV_SPEED / SWARM_SPEED scale.
        self.sim_env.SPEED_LIMIT = max(config.FPV_SPEED, config.SWARM_SPEED)

        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.agents = self.possible_agents[:]
        # Protected asset sits off the ground so the guardian rule doesn't dive the swarm into the floor.
        self.target_pos = np.array([0.0, 0.0, config.TARGET_ALTITUDE])
        self.max_steps = config.MAX_EPISODE_STEPS

        # Runtime state (also initialised in reset)
        self.step_count = 0
        self.current_wave = 1
        self.wave_delay_counter = 0
        self.fpv_active = True
        self.any_breach = False  # Survivor bonus is forfeited once the target is hit
        self.active_drones = list(range(self.num_drones))
        self.positions = np.zeros((self.num_drones, 3))
        self.velocities = np.zeros((self.num_drones, 3))
        self.fpv_pos = np.zeros(3)
        self.fpv_vel = np.zeros(3)
        self.prev_saw_fpv = [False] * self.num_drones
        # Camera forward direction per drone: last heading while moving (a hovering
        # drone's velocity direction is noise, so we keep the last meaningful one).
        self.headings = np.tile(np.array([1.0, 0.0, 0.0]), (self.num_drones, 1))
        # Per-drone short-term memory of the FPV (POMDP: with a limited FOV the
        # threat leaves the camera cone; the obs decays instead of vanishing).
        self.currently_sees = [False] * self.num_drones
        self.fpv_seen_age = np.full(self.num_drones, 10**9, dtype=np.int64)
        self.fpv_last_pos = np.zeros((self.num_drones, 3))
        self.fpv_last_vel = np.zeros((self.num_drones, 3))

    # ------------------------------------------------------------------
    def observation_space(self, agent: str) -> Box:
        return Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def action_space(self, agent: str) -> Box:
        return Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _spawn_swarm(self):
        """Teleport the swarm to a clustered spawn offset from the target.

        The swarm starts grouped away from the asset so the defensive formation
        (transit + spread) has to emerge, and the layout is re-randomized every
        episode (initial_xyzs is fixed at construction, so sim_env.reset() alone
        would replay the identical spawn forever).
        """
        angle = np.random.uniform(0, 2 * np.pi)
        center = self.target_pos + config.SWARM_SPAWN_DIST * np.array(
            [np.cos(angle), np.sin(angle), 0.0])

        spawn_positions = []
        for _ in range(self.num_drones):
            cand = None
            for _attempt in range(200):
                offset = np.random.uniform(-config.SWARM_SPAWN_RADIUS,
                                           config.SWARM_SPAWN_RADIUS, 2)
                z = np.random.uniform(config.SWARM_SPAWN_MIN_Z, config.SWARM_SPAWN_MAX_Z)
                cand = np.array([center[0] + offset[0], center[1] + offset[1], z])
                if all(np.linalg.norm(cand - q) >= config.SWARM_SPAWN_MIN_SPACING
                       for q in spawn_positions):
                    break
            spawn_positions.append(cand)  # accept last candidate if spacing failed

        client = self.sim_env.CLIENT
        for i, pos in enumerate(spawn_positions):
            drone_id = self.sim_env.DRONE_IDS[i]
            p.resetBasePositionAndOrientation(drone_id, pos.tolist(), [0, 0, 0, 1],
                                              physicsClientId=client)
            p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0],
                                physicsClientId=client)

        self.positions = np.array(spawn_positions)
        self.velocities = np.zeros((self.num_drones, 3))
        # Cameras start facing the asset they protect
        to_target = self.target_pos - self.positions
        self.headings = to_target / (np.linalg.norm(to_target, axis=1, keepdims=True) + 1e-8)

    # ------------------------------------------------------------------
    def _spawn_next_fpv(self):
        """Teleport the FPV drone to a new random approach position."""
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(config.FPV_SPAWN_MIN_DIST, config.FPV_SPAWN_MAX_DIST)  # inside DETECTION_RANGE
        height = np.random.uniform(config.FPV_SPAWN_MIN_Z, config.FPV_SPAWN_MAX_Z)          # at altitude, not on the floor

        new_x = self.target_pos[0] + distance * np.cos(angle)
        new_y = self.target_pos[1] + distance * np.sin(angle)
        new_pos = np.array([new_x, new_y, height])

        direction_to_target = self.target_pos - new_pos
        new_vel = (direction_to_target / np.linalg.norm(direction_to_target)) * config.FPV_SPEED

        fpv_id = self.sim_env.DRONE_IDS[self.num_drones]
        client = self.sim_env.CLIENT
        p.resetBasePositionAndOrientation(fpv_id, new_pos.tolist(), [0, 0, 0, 1],
                                          physicsClientId=client)
        p.resetBaseVelocity(fpv_id, new_vel.tolist(), [0, 0, 0],
                            physicsClientId=client)

        self.fpv_active = True
        self.fpv_pos = new_pos.copy()
        self.fpv_vel = new_vel.copy()
        # New threat: clear per-drone memories so stale tracks don't carry over,
        # and re-arm the scout bonus so first sight of EACH wave is rewarded.
        self.currently_sees = [False] * self.num_drones
        self.fpv_seen_age[:] = 10**9
        self.fpv_last_pos[:] = 0.0
        self.fpv_last_vel[:] = 0.0
        self.prev_saw_fpv = [False] * self.num_drones
        print(f"[ENV] Wave {self.current_wave} spawned at {new_pos}")

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.current_wave = 1
        self.wave_delay_counter = 0
        self.fpv_active = True
        self.any_breach = False
        self.active_drones = list(range(self.num_drones))
        self.prev_saw_fpv = [False] * self.num_drones  # NEW: Reset scout tracker

        obs_all, _ = self.sim_env.reset(seed=seed)
        self._update_internal_states(obs_all)
        # Clustered, randomized swarm spawn (overrides the fixed constructor layout)
        self._spawn_swarm()
        # Wave 1 spawns OUTSIDE detection range: the swarm has an unseen approach
        # phase in which to form up before the threat becomes visible.
        self._spawn_next_fpv()
        self._update_fpv_perception()

        infos = {agent: {} for agent in self.agents}
        return self._get_observations(), infos

    # ------------------------------------------------------------------
    def _update_internal_states(self, obs_all):
        """Read positions and velocities from the simulator observation."""
        self.positions = obs_all[:self.num_drones, 0:3].copy()
        self.velocities = obs_all[:self.num_drones, 10:13].copy()
        self.fpv_pos = obs_all[self.num_drones, 0:3].copy()
        self.fpv_vel = obs_all[self.num_drones, 10:13].copy()
        # Update camera headings from measured motion; keep the last heading while
        # hovering (velocity direction is numerical noise below HEADING_MIN_SPEED).
        speeds = np.linalg.norm(self.velocities, axis=1)
        moving = speeds > config.HEADING_MIN_SPEED
        self.headings[moving] = self.velocities[moving] / speeds[moving, None]

    # ------------------------------------------------------------------
    def _update_fpv_perception(self):
        """Per-drone FPV visibility + short-term memory of the last-seen state."""
        self.fpv_seen_age += 1
        self.currently_sees = [False] * self.num_drones
        if not self.fpv_active:
            return
        for i in self.active_drones:
            if is_visible(self.positions[i], self.headings[i], self.fpv_pos,
                          config.DETECTION_RANGE, config.CAMERA_FOV_DEGREES):
                self.currently_sees[i] = True
                self.fpv_seen_age[i] = 0
                self.fpv_last_pos[i] = self.fpv_pos
                self.fpv_last_vel[i] = self.fpv_vel

    # ------------------------------------------------------------------
    def step(self, actions):
        self.step_count += 1
        client = self.sim_env.CLIENT

        # Drones active at the START of this step (PettingZoo: rewards/terms only for these)
        active_at_start = list(self.active_drones)

        # 1. Build per-drone action array (zeros for inactive/missing agents)
        _zero = np.zeros(7, dtype=np.float32)
        action_array = np.array([actions.get(f"drone_{i}", _zero)
                                 for i in range(self.num_drones)])

        # 2. Compute desired swarm velocities via Boids controller
        desired_vels = compute_boids_velocity(
            self.positions, self.velocities, self.headings, self.fpv_pos,
            self.fpv_vel, self.target_pos, action_array
        )

        # 3. Compute FPV velocity
        vel_fpv = np.zeros(3)
        if self.fpv_active:
            dist_vec = self.target_pos - self.fpv_pos
            dist_norm = np.linalg.norm(dist_vec)
            if dist_norm > 1e-8:
                dir_to_target = dist_vec / dist_norm
                trip_dodge = any(
                    np.linalg.norm(self.fpv_pos - self.positions[i]) < config.FPV_DODGE_DIST
                    for i in self.active_drones
                )
                if trip_dodge:
                    random_vec = np.random.randn(3)
                    random_vec -= np.dot(random_vec, dir_to_target) * dir_to_target
                    random_vec /= (np.linalg.norm(random_vec) + 1e-8)
                    combined = dir_to_target + 0.5 * random_vec
                    combined /= (np.linalg.norm(combined) + 1e-8)
                    vel_fpv = combined * config.FPV_SPEED
                else:
                    vel_fpv = dir_to_target * config.FPV_SPEED

            # 4. Construct the action array for the simulator (Shape: N+1 x 4)
        # VelocityAviary expects [vx, vy, vz, yaw_rate]
        sim_actions = np.zeros((self.num_drones + 1, 4), dtype=np.float32)
        
        # Swarm gets Boids velocities, yaw = 0.
        # VelocityAviary normalizes [:3] to a direction and uses [3] as the speed
        # fraction of SPEED_LIMIT, so we must supply that fraction explicitly.
        sim_actions[:self.num_drones, :3] = desired_vels
        sim_actions[:self.num_drones, 3] = np.clip(
            np.linalg.norm(desired_vels, axis=1) / self.sim_env.SPEED_LIMIT, 0.0, 1.0)

        # FPV gets its computed velocity, or hovers if inactive
        if self.fpv_active:
            sim_actions[self.num_drones, :3] = vel_fpv
            sim_actions[self.num_drones, 3] = np.clip(
                np.linalg.norm(vel_fpv) / self.sim_env.SPEED_LIMIT, 0.0, 1.0)
        else:
            sim_actions[self.num_drones, :] = 0.0

        # 5. Advance physics (VelocityAviary's internal PID will smoothly track these velocities)
        obs_all, _, _, _, _ = self.sim_env.step(sim_actions)
            
        # 6. Update internal state. Keep the MEASURED velocities from the sim:
        # overwriting them with the commanded ones blinded the policy during dives
        # (the obs said "climbing" while the drone was actually falling).
        self._update_internal_states(obs_all)
        self._update_fpv_perception()

        # 7. Compute per-agent rewards and terminations for active_at_start drones only
        rewards = {f"drone_{i}": config.TIME_PEN for i in active_at_start}
        terminations = {f"drone_{i}": False for i in active_at_start}
        fpv_destroyed_this_step = False
        target_breached_this_step = False

        for i in active_at_start:
            agent_name = f"drone_{i}"

            # Ground collision
            if self.positions[i, 2] < config.GROUND_Z_THRESHOLD:
                rewards[agent_name] += config.SAC_PEN
                terminations[agent_name] = True
                if i in self.active_drones:
                    self.active_drones.remove(i)
                continue

            # FPV interaction
            if self.fpv_active:
                diff = self.fpv_pos - self.positions[i]
                dist_to_fpv = np.linalg.norm(diff)
                dir_to_fpv = diff / (dist_to_fpv + 1e-8)
                # RELATIVE closing speed: a drone parked in the FPV's path closes
                # at FPV speed. (Own-velocity-only made blocking — the optimal play
                # against a faster threat — register as a passive collision.)
                closing_speed = np.dot(self.velocities[i] - self.fpv_vel, dir_to_fpv)
                # Scouting: first sight of this wave earns the scout bonus
                if self.currently_sees[i] and not self.prev_saw_fpv[i]:
                    rewards[agent_name] += config.SCOUT_BONUS
                    self.prev_saw_fpv[i] = True  # Prevent farming this bonus

                if dist_to_fpv < config.COLLISION_DIST:
                    if closing_speed > config.RAMMING_THRESHOLD:
                        # Early interception earns a time-scaled bonus on top of base reward
                        time_fraction = 1.0 - (self.step_count / self.max_steps)
                        rewards[agent_name] += (config.SUCCESS_REWARD
                                                + time_fraction * config.EARLY_BONUS)
                        fpv_destroyed_this_step = True
                    else:
                        rewards[agent_name] += config.SAC_PEN
                        terminations[agent_name] = True
                        if i in self.active_drones:
                            self.active_drones.remove(i)
                elif closing_speed > 0:
                    # Dense shaped reward: small bonus for actively closing on FPV,
                    # normalized by the max possible relative closing speed
                    rewards[agent_name] += config.APPROACH_BONUS * (
                        closing_speed / (config.MAX_VEL_CMD + config.FPV_SPEED))

            # Idle-formation shaping (patrol band + teammate spacing): paid only
            # while this drone has NO fresh FPV track, so positioning incentives
            # never compete with an active engagement.
            if not self.fpv_active or self.fpv_seen_age[i] > config.FPV_MEMORY_STEPS:
                dist_t = np.linalg.norm(self.positions[i] - self.target_pos)
                if abs(dist_t - config.PATROL_RADIUS) < config.PATROL_WIDTH:
                    rewards[agent_name] += config.PATROL_BONUS
                d_nn = min((np.linalg.norm(self.positions[i] - self.positions[j])
                            for j in active_at_start if j != i),
                           default=config.SPACING_CAP)
                rewards[agent_name] += config.SPACING_BONUS * (
                    min(d_nn, config.SPACING_CAP) / config.SPACING_CAP)

            # Target breach
            if self.fpv_active and np.linalg.norm(self.fpv_pos - self.target_pos) < config.BLAST_RADIUS:
                rewards[agent_name] += config.FAIL_PEN
                target_breached_this_step = True

        # 8. Wave management
        if fpv_destroyed_this_step or target_breached_this_step:
            if fpv_destroyed_this_step:
                # Interception is a TEAM event: every drone still flying shares it,
                # not just the one that made contact (kills the kill lottery).
                for i in self.active_drones:
                    rewards[f"drone_{i}"] += config.TEAM_SUCCESS_REWARD
            if target_breached_this_step:
                self.any_breach = True
            self.fpv_active = False
            self.current_wave += 1

            if self.current_wave > self.max_waves or len(self.active_drones) == 0:
                # Episode over — survivors are rewarded only if the asset stayed safe:
                # staying alive must never pay while the target gets hit.
                if not self.any_breach:
                    for i in self.active_drones:
                        rewards[f"drone_{i}"] += config.SURVIVOR_REWARD
                for i in active_at_start:
                    terminations[f"drone_{i}"] = True
                self.active_drones = []
            else:
                self.wave_delay_counter = np.random.randint(
                    config.MIN_WAVE_DELAY_STEPS, config.MAX_WAVE_DELAY_STEPS
                )
        elif not self.fpv_active:
            self.wave_delay_counter -= 1
            if self.wave_delay_counter <= 0:
                self._spawn_next_fpv()

        # 9. Truncate on step limit
        if self.step_count >= self.max_steps:
            truncations = {f"drone_{i}": True for i in active_at_start}
            terminations = {f"drone_{i}": True for i in active_at_start}
            self.active_drones = []
        else:
            truncations = {f"drone_{i}": False for i in active_at_start}

        self.agents = [f"drone_{i}" for i in self.active_drones]
        # infos must be a subset of obs keys (Ray RLlib requirement)
        infos = {f"drone_{i}": {} for i in self.active_drones}
        return self._get_observations(), rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    def _get_observations(self):
        obs_dict = {}
        for i in self.active_drones:
            pos_i = self.positions[i]
            vel_i = self.velocities[i]

            # STRICTLY LOCAL + short memory: a drone gets FPV info only from ITS OWN
            # camera, but a track seen within the last FPV_MEMORY_STEPS persists with
            # a decaying freshness signal (1 = in sight now, ->0 = stale, 0 = blind).
            # Without this, a stateless policy forgets the threat the instant it
            # leaves the FOV, and zeroed rel_fpv is ambiguous with "FPV on top of me".
            age = self.fpv_seen_age[i]
            if self.fpv_active and age <= config.FPV_MEMORY_STEPS:
                rel_fpv_pos = self.fpv_last_pos[i] - pos_i
                rel_fpv_vel = self.fpv_last_vel[i] - vel_i
                freshness = 1.0 - age / config.FPV_MEMORY_STEPS
            else:
                rel_fpv_pos = np.zeros(3, dtype=np.float32)
                rel_fpv_vel = np.zeros(3, dtype=np.float32)
                freshness = 0.0

            rel_target_pos = self.target_pos - pos_i

            # Teammate summary (breaks the shared-policy symmetry: without it,
            # clustered drones get identical obs and act identically forever):
            # centre of mass of, and vector to the nearest of, living neighbours.
            neigh = [j for j in self.active_drones
                     if j != i and np.linalg.norm(self.positions[j] - pos_i)
                     < config.NEIGHBOR_RADIUS]
            if neigh:
                rel_com = np.mean(self.positions[neigh], axis=0) - pos_i
                nearest = min(neigh,
                              key=lambda j: np.linalg.norm(self.positions[j] - pos_i))
                rel_nearest = self.positions[nearest] - pos_i
                n_frac = len(neigh) / (self.num_drones - 1)
            else:
                rel_com = np.zeros(3, dtype=np.float32)
                rel_nearest = np.zeros(3, dtype=np.float32)
                n_frac = 0.0

            # 20D observation:
            # [own_vel (3), rel_fpv_pos (3), rel_fpv_vel (3), rel_target (3),
            #  freshness (1), rel_neighbor_com (3), rel_nearest_neighbor (3),
            #  neighbor_fraction (1)]
            obs = np.concatenate([vel_i, rel_fpv_pos, rel_fpv_vel, rel_target_pos,
                                  [freshness], rel_com, rel_nearest, [n_frac]])
            obs_dict[f"drone_{i}"] = obs.astype(np.float32)

        return obs_dict

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            self.sim_env.render()

    def close(self):
        try:
            self.sim_env.close()
        except Exception:
            pass
