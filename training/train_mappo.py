import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.swarm_env import SwarmInterceptEnv
from utils import config


def env_creator(env_config):
    """Factory function to create the environment for Ray RLlib."""
    render_mode = "human" if env_config.get("render", False) else None
    return SwarmInterceptEnv(render_mode=render_mode,
                             fov_degrees=env_config.get("fov_degrees"))


# FOV curriculum: start omnidirectional so the policy (re-)learns to intercept,
# then narrow toward the realistic forward camera. Jumping straight to a narrow
# FOV risks the "blind swarm, no intercept signal" failure mode.
# Iteration counts assume train_batch_size=16000 (4x the old 4000), so each
# iteration sees ~1333 env steps (~1.5 episodes) of 12-agent data. Phase 3 gets
# the most time: the narrow FOV is the hardest setting and the one that ships.
CURRICULUM = [
    # (fov_degrees, iterations)
    (360.0, 80),
    (180.0, 80),
    (120.0, 160),
]


def build_ppo(env_name, fov_degrees):
    ppo_config = (
        PPOConfig()
        # PettingZoo's Dict obs space requires the old API stack in Ray 2.x
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name,
                     env_config={"render": False, "fov_degrees": fov_degrees})
        .framework("torch")
        .env_runners(
            num_env_runners=2,               # Use 2 parallel envs for diverse experience
            num_envs_per_env_runner=1,
            rollout_fragment_length=200,
            batch_mode="truncate_episodes",  # CRITICAL FIX: Prevents Ray from hanging
        )
        .training(
            train_batch_size=16000,  # 4000 was ~330 env steps (~1/4 episode) -> noisy updates
            minibatch_size=512,
            num_epochs=10,
            lr=3e-4,
            gamma=0.995,  # 1200-step episodes: terminal rewards must survive discounting
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.02,     # a touch more exploration early on
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN")
    )
    return ppo_config.build()


def main():
    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=0)
    env_name = "swarm_intercept_v0"
    register_env(env_name, lambda cfg: ParallelPettingZooEnv(env_creator(cfg)))

    print("Starting MAPPO Training (FOV curriculum)...", flush=True)
    weights = None
    total_iter = 0

    for phase, (fov, iterations) in enumerate(CURRICULUM, start=1):
        print(f"\n=== Phase {phase}/{len(CURRICULUM)}: FOV {fov:.0f} deg, "
              f"{iterations} iterations ===", flush=True)
        algo = build_ppo(env_name, fov)

        # Carry the policy over from the previous phase (same obs/action spaces,
        # only the env's FOV changed, so the weights transfer directly).
        if weights is not None:
            algo.set_weights(weights)
            try:
                algo.env_runner_group.sync_weights()
            except AttributeError:
                algo.workers.sync_weights()  # older Ray attribute name

        for _ in range(iterations):
            result = algo.train()
            total_iter += 1

            if total_iter % 5 == 0:
                env_runners = result.get("env_runners", {})
                episode_reward = env_runners.get("episode_reward_mean", float("nan"))
                episode_len = env_runners.get("episode_len_mean", float("nan"))
                num_eps = env_runners.get("num_episodes", 0)
                print(f"Phase {phase} (FOV {fov:.0f}) | Iteration {total_iter:3d} | "
                      f"Mean Reward: {episode_reward:8.2f} | "
                      f"Mean Length: {episode_len:5.1f} | Episodes: {num_eps}", flush=True)

            if total_iter % 25 == 0:
                checkpoint = algo.save(f"./checkpoints/iter_{total_iter}")
                print(f"Checkpoint saved at {checkpoint}", flush=True)

        weights = algo.get_weights()
        checkpoint = algo.save(f"./checkpoints/phase{phase}_fov{int(fov)}")
        print(f"Phase {phase} complete. Checkpoint: {checkpoint}", flush=True)

        if phase == len(CURRICULUM):
            final_checkpoint = algo.save("./checkpoints/final_model")
            print(f"Training complete! Final model saved at {final_checkpoint}", flush=True)
        algo.stop()

    ray.shutdown()


if __name__ == "__main__":
    main()
