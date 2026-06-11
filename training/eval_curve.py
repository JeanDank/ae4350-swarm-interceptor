"""
Quantitatively evaluate saved checkpoints headlessly and print a learning-progression
table (mean reward, interceptions, survivors, length). Since the training script logs
metrics only to the console, this reconstructs "the result" from the checkpoints.

Usage:
    python training/eval_curve.py                      # all checkpoints/iter_* + final_model
    python training/eval_curve.py --episodes 10        # episodes per checkpoint (default 8)
    python training/eval_curve.py checkpoints/iter_200 # just one checkpoint
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.swarm_env import SwarmInterceptEnv
from utils import config
from training.visualize import run_and_log

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover_checkpoints(arg):
    if arg:
        return [arg]
    ck = os.path.join(_ROOT, "checkpoints")
    iters = sorted(glob.glob(os.path.join(ck, "iter_*")),
                   key=lambda p: int(p.split("_")[-1]))
    final = os.path.join(ck, "final_model")
    if os.path.isdir(final):
        iters.append(final)
    return iters


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints into a results table")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()

    checkpoints = discover_checkpoints(args.checkpoint)
    if not checkpoints:
        print("No checkpoints found in ./checkpoints/. Train first.")
        sys.exit(1)

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=0, logging_level="ERROR")
    register_env("swarm_intercept_v0",
                 lambda cfg: ParallelPettingZooEnv(SwarmInterceptEnv()))

    env = SwarmInterceptEnv(render_mode=None)
    print(f"\nEvaluating {len(checkpoints)} checkpoint(s), {args.episodes} episodes each "
          f"(max {config.NUM_FPV_WAVES} intercepts/episode)\n")
    print(f"{'checkpoint':<16}{'reward':>12}{'intercepts':>14}{'survivors':>12}{'len':>8}")
    print("-" * 62)

    try:
        for path in checkpoints:
            algo = Algorithm.from_checkpoint(path)
            R, K, S, L = [], [], [], []
            for _ in range(args.episodes):
                log, total = run_and_log(env, algo=algo)
                R.append(total)
                K.append(len(log["intercepts"]))
                S.append(int(log["active"][-1].sum()))
                L.append(len(log["positions"]))
            algo.stop()
            name = os.path.basename(path.rstrip("/\\"))
            print(f"{name:<16}{np.mean(R):>12.0f}{np.mean(K):>14.2f}"
                  f"{np.mean(S):>12.1f}{np.mean(L):>8.0f}")
    finally:
        env.close()
        ray.shutdown()
    print("\n(reward = sum over all agents; intercepts out of "
          f"{config.NUM_FPV_WAVES}; survivors out of {config.NUM_DRONES})")


if __name__ == "__main__":
    main()
