"""
Render a trained swarm-interceptor checkpoint to a saved 3D animation (MP4/GIF).

Runs entirely headless (no PyBullet GUI) and writes a shareable animation showing
the protected target, the swarm interceptors, the FPV threat, and a flash at each
interception. Useful for the assignment report.

Usage:
    python training/visualize.py                              # final_model -> results/intercept.mp4
    python training/visualize.py checkpoints/iter_100         # specific checkpoint
    python training/visualize.py --random                     # random policy (no checkpoint needed)
    python training/visualize.py checkpoints/iter_100 --episodes 5 --out results/best.gif --fps 25

By default it runs several episodes and animates the best (highest-reward) one, so the
clip shows a representative successful intercept rather than the first random outcome.
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.swarm_env import SwarmInterceptEnv
from utils import config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "final_model")


# --------------------------------------------------------------------------- #
def load_policy(checkpoint_path):
    """Load an RLlib Algorithm from a checkpoint (old API stack)."""
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    # Restoring a checkpoint saved with num_env_runners>0 spawns worker subprocesses
    # that must `import environment`. Ray's local workers inherit the driver's env, so
    # prepend the project root to PYTHONPATH here -> import works from any CWD.
    os.environ["PYTHONPATH"] = _ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    # The checkpoint was trained with num_env_runners=2, so restoring spawns 2 worker
    # actors (each needs a CPU). num_cpus must cover them or Ray deadlocks waiting.
    ray.init(ignore_reinit_error=True, num_cpus=3, num_gpus=0, logging_level="ERROR")
    register_env("swarm_intercept_v0",
                 lambda cfg: ParallelPettingZooEnv(
                     SwarmInterceptEnv(fov_degrees=cfg.get("fov_degrees"))))
    # pyarrow rejects relative paths ("URI has empty scheme") -> use absolute.
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    print("Checkpoint loaded.")
    return algo


# --------------------------------------------------------------------------- #
def run_and_log(env, algo=None):
    """Run one headless episode, returning a per-step trajectory log and total reward."""
    obs, _ = env.reset()
    P, A, Fp, FA, Wv, intercepts = [], [], [], [], [], []
    total, step = 0.0, 0
    done = False

    while not done:
        actions = {}
        for aid, o in obs.items():
            if algo is not None:
                actions[aid] = algo.compute_single_action(
                    o, policy_id="default_policy", explore=False)
            else:
                actions[aid] = env.action_space(aid).sample()

        obs, rewards, term, trunc, _ = env.step(actions)
        step += 1
        total += sum(rewards.values())

        # Record state after the step
        P.append(env.positions.copy())
        active = np.zeros(env.num_drones, dtype=bool)
        for i in env.active_drones:
            active[i] = True
        A.append(active)
        Fp.append(env.fpv_pos.copy())
        FA.append(bool(env.fpv_active))
        Wv.append(env.current_wave)
        for r in rewards.values():
            if r >= config.SUCCESS_REWARD:
                intercepts.append((step - 1, env.fpv_pos.copy()))

        done = (not env.agents) or (term and all(term.values())) \
            or (trunc and all(trunc.values()))
        if step > config.MAX_EPISODE_STEPS + 2:
            break

    log = dict(
        positions=np.array(P), active=np.array(A), fpv_pos=np.array(Fp),
        fpv_active=np.array(FA), wave=np.array(Wv),
        target=env.target_pos.copy(), intercepts=intercepts,
    )
    return log, total


# --------------------------------------------------------------------------- #
def animate(log, out_path, fps, stride, title_prefix):
    """Build and save a 3D animation from a trajectory log."""
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    P, A = log["positions"], log["active"]
    F, FA, W = log["fpv_pos"], log["fpv_active"], log["wave"]
    target, intercepts = log["target"], dict(log["intercepts"])
    T, N, _ = P.shape
    frames = list(range(0, T, max(1, stride)))
    R = config.BLAST_RADIUS
    trail = 10

    # Axis bounds from all data
    pts = np.concatenate([P.reshape(-1, 3), F, target[None, :]], axis=0)
    lo, hi = pts.min(axis=0) - 2.0, pts.max(axis=0) + 2.0

    # Pre-compute a low-res wireframe sphere for the protected blast radius
    u = np.linspace(0, 2 * np.pi, 14)
    v = np.linspace(0, np.pi, 9)
    sx = target[0] + R * np.outer(np.cos(u), np.sin(v))
    sy = target[1] + R * np.outer(np.sin(u), np.sin(v))
    sz = target[2] + R * np.outer(np.ones_like(u), np.cos(v))

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="k", markersize=15, label="Protected target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markersize=8, label="Swarm interceptors"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="tab:red",
               markersize=10, label="FPV threat"),
    ]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def seg_start(t):
        s = t
        while s > 0 and W[s - 1] == W[t]:
            s -= 1
        return s

    def update(t):
        ax.cla()
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(max(0.0, lo[2]), hi[2])
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")

        # Protected target + blast-radius sphere + ground hint
        ax.plot_wireframe(sx, sy, sz, color="gold", alpha=0.12, linewidth=0.5)
        ax.scatter(*target, c="gold", marker="*", s=320,
                   edgecolors="k", depthshade=False)

        s0 = seg_start(t)
        t0 = max(s0, t - trail)
        # Swarm: full trajectory lines + current positions (only living drones)
        for i in range(N):
            if A[t, i]:
                tr = P[:t + 1, i, :]
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], c="tab:blue", alpha=0.30, lw=1)
                ax.scatter(P[t, i, 0], P[t, i, 1], P[t, i, 2],
                           c="tab:blue", s=32, depthshade=False)
        # FPV threat: trail + current position
        if FA[t]:
            tr = F[t0:t + 1]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], c="tab:red", alpha=0.5, lw=1.5)
            ax.scatter(F[t, 0], F[t, 1], F[t, 2], c="tab:red", s=70,
                       marker="^", depthshade=False)
        # Interception flash (decays over a few frames)
        for it, ipos in intercepts.items():
            age = t - it
            if 0 <= age < 6:
                ax.scatter(ipos[0], ipos[1], ipos[2], c="yellow",
                           s=450 * (1 - age / 6), marker="*",
                           edgecolors="orange", depthshade=False, alpha=0.9)

        n_alive = int(A[t].sum())
        n_kills = sum(1 for it in intercepts if it <= t)
        ax.set_title(f"{title_prefix}   step {t}/{T - 1}   wave {int(W[t])}   "
                     f"alive {n_alive}/{N}   intercepts {n_kills}")
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / fps, blit=False)

    # Choose a writer: MP4 via ffmpeg if available, else fall back to GIF
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.lower().endswith(".mp4") and not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH -> saving GIF instead.")
        out_path = out_path[:-4] + ".gif"
    if out_path.lower().endswith(".mp4"):
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    else:
        writer = PillowWriter(fps=fps)

    print(f"Rendering {len(frames)} frames -> {out_path} ...")
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved animation: {out_path}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Render swarm intercept to a 3D animation file")
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT,
                        help="Checkpoint dir (default: checkpoints/final_model)")
    parser.add_argument("--random", action="store_true",
                        help="Use a random policy (no checkpoint needed)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Run N episodes and animate the best one (default: 3)")
    parser.add_argument("--out", default=os.path.join(_ROOT, "results", "intercept.mp4"),
                        help="Output file (.mp4 or .gif). Default: results/intercept.mp4")
    parser.add_argument("--fps", type=int, default=20, help="Animation frame rate")
    parser.add_argument("--stride", type=int, default=1,
                        help="Animate every Nth sim step (>=2 for shorter/lighter clips)")
    parser.add_argument("--fov", type=float, default=None,
                        help="Camera FOV in degrees (default: config value; "
                             "match the curriculum phase of the checkpoint)")
    args = parser.parse_args()

    algo = None
    if not args.random:
        ckpt = args.checkpoint
        if not os.path.isdir(ckpt):
            # Fall back to a path relative to the project root, so it works from any CWD
            alt = os.path.join(_ROOT, args.checkpoint)
            if os.path.isdir(alt):
                ckpt = alt
            else:
                print(f"Checkpoint not found: {args.checkpoint}")
                print(f"(also tried: {alt})")
                print("Pass a valid checkpoint dir, or use --random.")
                sys.exit(1)
        algo = load_policy(ckpt)
    else:
        print("Using RANDOM policy (no checkpoint).")

    env = SwarmInterceptEnv(render_mode=None, fov_degrees=args.fov)  # headless
    try:
        best_log, best_r = None, -np.inf
        for ep in range(1, args.episodes + 1):
            log, r = run_and_log(env, algo=algo)
            n_kills = len(log["intercepts"])
            print(f"  Episode {ep}: {len(log['positions'])} steps | "
                  f"reward {r:.0f} | intercepts {n_kills}")
            if r > best_r:
                best_log, best_r = log, r
    finally:
        env.close()
        if algo is not None:
            import ray
            algo.stop()
            ray.shutdown()

    label = "RANDOM policy" if args.random else os.path.basename(args.checkpoint.rstrip("/\\"))
    print(f"\nAnimating best episode (reward {best_r:.0f})...")
    animate(best_log, args.out, args.fps, args.stride,
            title_prefix=f"Swarm intercept — {label}")


if __name__ == "__main__":
    main()
