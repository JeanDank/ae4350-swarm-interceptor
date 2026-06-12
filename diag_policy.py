"""Throwaway diagnostic: what does the trained policy actually do? (deleted after use)"""
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.swarm_env import SwarmInterceptEnv
from training.visualize import load_policy
from utils import config

EPISODES = 6
WEIGHT_NAMES = ["w_pred", "w_react", "w_pres", "w_coh", "w_sep", "w_guard", "w_orbit"]


def main():
    algo = load_policy(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "checkpoints", "final_model"))
    env = SwarmInterceptEnv()

    all_weights = []          # every action vector the policy emitted
    dists_to_target = []      # radial distance of living drones, every step
    nn_dists = []             # nearest-neighbour distance, every step
    death_log = []            # (episode, step, cause, peak_z_before_death)
    intercepts = breaches = 0
    z_hist = {}               # drone -> recent altitudes

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        z_hist = {i: [] for i in range(env.num_drones)}
        done, step = False, 0
        while not done:
            actions = {aid: algo.compute_single_action(o, policy_id="default_policy",
                                                       explore=False)
                       for aid, o in obs.items()}
            space = env.action_space("drone_0")
            all_weights.extend(np.clip(a, space.low, space.high)
                               for a in actions.values())
            before = set(env.active_drones)

            obs, rewards, term, trunc, _ = env.step(actions)
            step += 1

            for i in env.active_drones:
                z_hist[i].append(env.positions[i, 2])
                dists_to_target.append(np.linalg.norm(env.positions[i] - env.target_pos))
                others = [j for j in env.active_drones if j != i]
                if others:
                    nn_dists.append(min(np.linalg.norm(env.positions[i] - env.positions[j])
                                        for j in others))

            for r in rewards.values():
                if r >= config.SUCCESS_REWARD:
                    intercepts += 1
                if r <= config.FAIL_PEN and r > config.SAC_PEN:
                    pass  # breach counted once below

            dead = before - set(env.active_drones)
            episode_over = (not env.agents)
            for i in dead:
                if episode_over and step >= config.MAX_EPISODE_STEPS:
                    continue  # truncation, not a death
                if env.positions[i, 2] < config.GROUND_Z_THRESHOLD:
                    cause = "ground"
                elif np.linalg.norm(env.fpv_pos - env.positions[i]) < config.COLLISION_DIST * 2:
                    cause = "passive-collision"
                else:
                    cause = "episode-end"
                peak = max(z_hist[i][-150:]) if z_hist[i] else float("nan")
                death_log.append((ep, step, cause, peak))

            done = (not env.agents) or (term and all(term.values())) \
                or (trunc and all(trunc.values()))
            if step > config.MAX_EPISODE_STEPS + 2:
                break
        if env.any_breach:
            breaches += 1
        print(f"ep {ep}: {step} steps, survivors {len(env.active_drones)}, "
              f"breach={env.any_breach}")

    env.close()
    import ray
    algo.stop()
    ray.shutdown()

    W = np.array(all_weights)
    D = np.array(dists_to_target)
    NN = np.array(nn_dists)
    print("\n=== policy action weights (mean +- std over all steps/drones) ===")
    for k, name in enumerate(WEIGHT_NAMES):
        print(f"  {name:8s}: {W[:, k].mean():.3f} +- {W[:, k].std():.3f}")
    print("\n=== radial distance to target (living drones, all steps) ===")
    print(f"  p10/p50/p90: {np.percentile(D, 10):.1f} / {np.percentile(D, 50):.1f} / "
          f"{np.percentile(D, 90):.1f} m   (patrol shell is at {config.PATROL_RADIUS} m)")
    print(f"  fraction inside band |d-10|<4: {np.mean(np.abs(D - 10) < 4):.2f}")
    print("\n=== nearest-neighbour distance ===")
    print(f"  p10/p50/p90: {np.percentile(NN, 10):.1f} / {np.percentile(NN, 50):.1f} / "
          f"{np.percentile(NN, 90):.1f} m")
    print(f"\n=== deaths ({len(death_log)} total over {EPISODES} eps) ===")
    for ep, step, cause, peak in death_log:
        print(f"  ep {ep} step {step:4d}: {cause:18s} peak z in last ~3s: {peak:.1f} m")
    print(f"\nintercept events: {intercepts} | episodes with breach: {breaches}/{EPISODES}")


if __name__ == "__main__":
    main()
