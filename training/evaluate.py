"""
Evaluate a trained MAPPO checkpoint in the PyBullet 3D GUI.

Usage:
    python training/evaluate.py                          # loads final_model
    python training/evaluate.py checkpoints/iter_300    # specific checkpoint
    python training/evaluate.py --random                # random policy (no checkpoint)
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.swarm_env import SwarmInterceptEnv
from utils import config

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "final_model"
)


def load_policy(checkpoint_path):
    """Load an RLlib Algorithm from a checkpoint."""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig

    # Restoring a checkpoint saved with num_env_runners>0 spawns worker subprocesses
    # that must `import environment`. Ray's local workers inherit the driver's env, so
    # prepend the project root to PYTHONPATH here -> import works from any CWD.
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["PYTHONPATH"] = _root + os.pathsep + os.environ.get("PYTHONPATH", "")
    # The checkpoint was trained with num_env_runners=2, so restoring spawns 2 worker
    # actors (each needs a CPU). num_cpus must cover them or Ray deadlocks waiting.
    ray.init(ignore_reinit_error=True, num_cpus=3, num_gpus=0, logging_level="ERROR")

    # Re-register env so RLlib can validate obs/action spaces (honor the FOV the
    # checkpoint was trained with, stored in its env_config)
    register_env("swarm_intercept_v0",
                 lambda cfg: ParallelPettingZooEnv(
                     SwarmInterceptEnv(fov_degrees=cfg.get("fov_degrees"))))

    # pyarrow (used by Ray's from_checkpoint) rejects relative paths with
    # "URI has empty scheme" -> always pass an absolute path.
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Loading checkpoint from: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("Checkpoint loaded.")
    return algo


def run_episode(env, algo=None, episode_num=1):
    """Run one episode. Uses trained policy if algo provided, else random."""
    obs_dict, _ = env.reset()
    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    step = 0
    episode_done = False

    while not episode_done:
        step += 1

        # Detect GUI window closed (PyBullet disconnects when user shuts the window)
        import pybullet as p
        if not p.isConnected(env.sim_env.CLIENT):
            print("\n[PyBullet window closed — stopping.]")
            break

        actions = {}
        for agent_id, obs in obs_dict.items():
            if algo is not None:
                actions[agent_id] = algo.compute_single_action(
                    obs, policy_id="default_policy", explore=False
                )
            else:
                actions[agent_id] = env.action_space(agent_id).sample()

        try:
            obs_dict, rewards, terminations, truncations, _ = env.step(actions)
        except Exception as e:
            if "connected" in str(e).lower():
                print("\n[PyBullet window closed — stopping.]")
                break
            raise

        for agent_id, r in rewards.items():
            total_rewards[agent_id] = total_rewards.get(agent_id, 0.0) + r

        all_done = (not env.agents) or all(terminations.values()) or all(truncations.values())
        episode_done = all_done

        # Throttle to ~30 fps so you can actually watch what's happening
        time.sleep(1.0 / 30.0)

    total = sum(total_rewards.values())
    per_agent = total / config.NUM_DRONES
    survivors = len(env.active_drones)
    print(f"  Episode {episode_num}: {step} steps | "
          f"Total reward: {total:.0f} | Per-agent: {per_agent:.1f} | "
          f"Survivors: {survivors}/{config.NUM_DRONES}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Evaluate swarm interceptor in PyBullet GUI")
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint directory (default: checkpoints/final_model)")
    parser.add_argument("--random", action="store_true",
                        help="Use random policy instead of loading a checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Camera FOV in degrees (default: config value; "
                             "match the curriculum phase of the checkpoint)")
    args = parser.parse_args()

    # Load policy
    algo = None
    if not args.random:
        if not os.path.isdir(args.checkpoint):
            # Fall back to a path relative to the project root, so it works from any CWD
            alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               args.checkpoint)
            if os.path.isdir(alt):
                args.checkpoint = alt
            else:
                print(f"Checkpoint not found: {args.checkpoint}")
                print(f"(also tried: {alt})")
                print("Use --random to run without a checkpoint, or specify a valid path.")
                sys.exit(1)
        algo = load_policy(args.checkpoint)
    else:
        print("Running with RANDOM policy (no checkpoint).")

    # Create env with GUI
    print("Opening PyBullet GUI...")
    env = SwarmInterceptEnv(render_mode="human", fov_degrees=args.fov)

    print(f"\nRunning {args.episodes} episode(s) — close the PyBullet window to stop early.\n")
    rewards = []
    try:
        for ep in range(1, args.episodes + 1):
            import pybullet as p
            if not p.isConnected(env.sim_env.CLIENT):
                print("PyBullet window closed.")
                break
            r = run_episode(env, algo=algo, episode_num=ep)
            rewards.append(r)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()
        if algo is not None:
            import ray
            algo.stop()
            ray.shutdown()

    if rewards:
        print(f"\n--- Summary over {len(rewards)} episode(s) ---")
        print(f"  Mean total reward : {np.mean(rewards):.1f}")
        print(f"  Best              : {np.max(rewards):.1f}")
        print(f"  Worst             : {np.min(rewards):.1f}")


if __name__ == "__main__":
    main()
