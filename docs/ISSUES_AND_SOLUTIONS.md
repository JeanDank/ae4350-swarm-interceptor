# Issues Encountered & Solutions

A chronological log of the problems hit while building the swarm-interceptor
(AE4350) and how each was solved. Useful as a reference for the report.

---

## 1. Simulation / physics (gym-pybullet-drones)

**Everything hovered in place, swarm and FPV frozen.**
- *Cause:* `VelocityAviary` actions are `[dir_x, dir_y, dir_z, speed_fraction]`. The 4th
  component was hardcoded to `0.0`, which means "0% of speed limit".
- *Fix:* set it to `clip(‖v_desired‖ / SPEED_LIMIT, 0, 1)`. Also raise the default
  `SPEED_LIMIT` (~0.25 m/s) after construction — it is far below mission speeds.

**Drones tilted over and crashed when commanded fast.**
- *Cause:* the CF2X can only hold altitude up to ~2.5 m/s; commanding 8 m/s tilts it so
  hard it sinks into the ground.
- *Fix:* keep all speeds ≤ 2.5 m/s (`FPV_SPEED = 2.5`, `SWARM_SPEED = 2.0`).

**Drones slowly sank during fast horizontal flight.**
- *Cause:* `ctrl_freq = 10 Hz` — the velocity PID only corrected every 0.1 s.
- *Fix:* `ctrl_freq = 48 Hz` (config.CTRL_FREQ). Holds altitude stably.

**Whole swarm dove into the floor (240/240 ground deaths).**
- *Cause:* the protected target sat at z = 0, so the guardian rule pointed into the ground.
- *Fix:* raise the target to z = 3 (`TARGET_ALTITUDE`) and add a ground-avoidance reflex in
  the Boids controller: below `GROUND_SAFE_Z = 2.5` suppress horizontal velocity and force a
  climb, both scaling with the altitude deficit. Ground deaths dropped to ~9/240.

## 2. Reward gradient & learning signal (2026-06-08)

**Agent learned to suicide immediately (reward −632, episode length 40).**
- *Cause:* `TIME_PEN = −1` dominated the tiny approach bonus, so ending the episode fast
  was optimal; `FAIL_PEN = −1000` added huge return variance.
- *Fix:* `TIME_PEN → −0.02`, `APPROACH_BONUS 0.05 → 1.0`, `FAIL_PEN −1000 → −200`.

**No intercept signal at all.**
- *Cause:* FPV spawned 42 m away with a 20 m detection range — the swarm was blind.
- *Fix:* spawn the FPV within sensing reach (later reworked: spawn outside detection range
  but add a scout bonus and an FOV curriculum, see §3).

**Unstable PPO updates.**
- *Cause:* `train_batch_size = 400` was less than one episode of 12-agent data.
- *Fix:* `4000` (later `16000`, see §4).

## 3. Emergent behaviour rework (2026-06-10)

**Swarm tail-chased the FPV instead of blocking it.**
- *Fix:* invert the speeds (FPV 2.5 m/s > swarm 2.0 m/s) so pursuit can never close, and
  change the kill check to RELATIVE closing speed `dot(v_drone − v_fpv, dir_to_fpv)` so
  parking in the FPV's path counts as an interception instead of a passive collision.

**Stateless policy forgot the FPV the instant it left the camera FOV.**
- *Fix:* per-drone short-term memory: last-seen FPV position/velocity persists in the obs
  for `FPV_MEMORY_STEPS = 48` with a decaying freshness flag (1 = in sight, 0 = blind).

**Hovering drones had garbage camera headings.**
- *Cause:* a hovering drone's velocity direction is numerical noise.
- *Fix:* maintain per-drone headings, updated only while speed > 0.3 m/s.

**Every episode replayed the identical spawn layout.**
- *Cause:* `initial_xyzs` is fixed at construction; `sim_env.reset()` replays it.
- *Fix:* teleport-based randomized clustered spawn (`_spawn_swarm`) in `reset()`.

**Narrow FOV from scratch = blind swarm, no learning signal.**
- *Fix:* FOV curriculum in training: 360° → 180° → 120°, weights carried between phases.

## 4. Final-model behaviour fixes (2026-06-11)

**Drones deliberately suicided by climbing high, then diving into the ground.**
- *Cause:* bad death economics. Each breach cost every living drone −200 (×2 waves), while
  dying cost only −50 once — and the +500 kill reward went to a single drone (a lottery for
  the other 11). Climb-then-dive was the learned trick to build enough downward momentum to
  defeat the ground-avoidance reflex.
- *Fix:* `SAC_PEN −50 → −400` (death is never cheaper than fighting on);
  `TEAM_SUCCESS_REWARD = 300` to every living drone per interception (killer keeps
  +500 + early bonus on top), so protecting the target dominates everything;
  `SURVIVOR_REWARD 20 → 100` but paid only if the target was never breached.

**Swarm collapsed into one tight cluster on the target; no spreading, no scouting.**
- *Cause:* the guardian rule was a point attractor and weights are clamped to [0, 1], so
  nothing could push outward except the 1.5 m separation rule — the cluster was the only
  reachable equilibrium. Drones also had zero teammate information in the obs, so the
  shared policy produced identical actions (symmetry never broke).
- *Fix:*
  - Guardian became a patrol *band*: `tanh((dist − PATROL_RADIUS 10 m) / 4 m)` along the
    radial — repels inside, attracts outside, equilibrium is a shell, not a point.
  - New 7th Boids weight: tangential *orbit* rule. Since the camera heading follows the
    velocity, orbiting doubles as scanning (fixes scouting).
  - Obs 13D → 20D: + relative neighbour centre-of-mass, vector to nearest neighbour, and
    neighbour fraction (within 15 m) — breaks the symmetry.
  - Small idle-time shaping (only while a drone has no fresh FPV track): +0.02/step for
    holding the patrol band and +0.02/step scaled by nearest-neighbour distance (cap 6 m).
  - `SEPARATION_RADIUS` stays 1.5 m: it is a minimum-spacing constraint, not a spreading
    mechanism.

**Value function couldn't see crashes coming.**
- *Cause:* `step()` overwrote measured velocities with commanded ones, so during a dive
  through the reflex zone the obs said "climbing" while the drone was falling.
- *Fix:* keep the measured velocities from the simulator.

**Training knobs for 1200-step episodes.**
- `gamma 0.99 → 0.995` (terminal rewards must survive discounting),
  `train_batch_size 4000 → 16000`, `minibatch 256 → 512`,
  curriculum iterations rescaled 150/150/200 → 60/60/80 (≈1.6× total experience).
- Note: these changes made all earlier checkpoints (incl. `final_model`) incompatible —
  retrain from scratch.

## 5. Emergence rework (2026-06-12)

A 6-episode diagnostic of the retrained model (72 drone-lives) showed the §4 reward fixes
were necessary but not sufficient: every drone still died every episode (mostly ground),
every episode was breached, and only 4/12 waves were intercepted. The patrol band *did*
hold (82% of drone-time inside it) — but as one tight clump (median neighbour distance
1.5 m), not a ring.

**Drones still crashed into the ground — but from cruise altitude, not high dives.**
- *Cause:* no longer reward-driven (death costs −400). The Boids output was normalized to
  full speed (`normalize(combined) * MAX_VEL_CMD`), so a drone chasing a low FPV dove at
  2 m/s and built momentum the ground reflex could not reverse within 2.5 m.
- *Fix:* (1) variable speed — **cap** the combined magnitude instead of normalizing
  (`speed = MAX_VEL_CMD · min(‖combined‖, 1)`), so drones can slow down, hover, park, and
  block; (2) a **descent brake** above the reflex zone: commanded sink rate is limited to
  `(z − GROUND_SAFE_Z)/GROUND_BRAKE_TIME`, making dive-throughs impossible from any height.

**Swarm held the shell as a single clump; no dispersal, no emergent formation.**
- *Cause:* full-speed normalization made differential speeds (needed for a clump to
  stretch) impossible, and the orbit rule was hardcoded counterclockwise, so even heavy
  orbiting (w_orbit ≈ 0.74) moved the clump as one body.
- *Fix (emergence over scripting):*
  - **Signed orbit weight**: action component 7 is now in [−1, 1] — each drone *chooses*
    its orbit direction from its own observations, instead of a hardcoded split.
  - **Own heading added to the obs** (23D): without the camera-forward vector the policy
    cannot relate neighbours/threats to its FOV or make left/right decisions — both
    prerequisites for scanning and dispersal to emerge.
  - **Angular-coverage shaping replaces metric spacing**: the idle bonus now pays for
    bearing separation around the target (saturating at the perfect-ring share 2π/N).
    This rewards the *goal* — no approach direction left undefended — while leaving the
    mechanism (ring, sectors, scout splits) free to emerge.
  - Idle shaping raised 0.02 → 0.05/step so the formation gradient is visible to PPO.
- Curriculum lengthened to 80/80/160 iterations (phase 3 — the narrow-FOV setting that
  actually ships — gets the most time).

## 6. Ray / RLlib / PettingZoo quirks

- Ray 2.55 needs the **old API stack** for PettingZoo Dict obs spaces:
  `enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False`.
- `batch_mode="complete_episodes"` hung Ray → use `"truncate_episodes"`.
- Episode metrics live at `result["env_runners"]["episode_reward_mean"]`, not top-level.
- PettingZoo 1.26: `observation_space(agent)` / `action_space(agent)` are methods; do not
  call `super().reset()`.
- Restoring a checkpoint spawns worker subprocesses: prepend the project root to
  `PYTHONPATH` and pass an **absolute** checkpoint path (pyarrow rejects relative ones),
  and give `ray.init` enough CPUs to cover the saved `num_env_runners`.
