#!/home/plague/pyenv_ros/bin/python3

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_20_rewards = []

    def _on_training_start(self) -> None:
        # Initialize arrays here to ensure self.model is available
        self.current_rewards = np.zeros(self.training_env.num_envs)
        self.current_lengths = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        self.current_rewards += rewards
        self.current_lengths += 1

        for i in range(len(dones)):
            if dones[i]:
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lengths[i])
                self.last_20_rewards.append(self.current_rewards[i])

                if len(self.last_20_rewards) > 20:
                    self.last_20_rewards.pop(0)

                # Print current episode's reward and length
                # print(f"Episode {len(self.episode_rewards)} - Reward: {self.current_rewards[i]}, Length: {self.current_lengths[i]}")

                self.current_rewards[i] = 0
                self.current_lengths[i] = 0

        # Debugging: Print rewards and lengths at each step
        # print(f"Step {self.num_timesteps} - Rewards: {rewards}, Dones: {dones}")
        # print(f"Current rewards: {self.current_rewards}, Current lengths: {self.current_lengths}")

        return True

    def _on_rollout_end(self) -> None:
        mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        mean_reward_last_20 = np.mean(self.last_20_rewards[-20:]) if len(self.last_20_rewards) >= 20 else 0

        # Print the average metrics
        # print(f"Mean Reward: {mean_reward}, Mean Length: {mean_length}, Mean Reward Last 20: {mean_reward_last_20}")

        self.logger.record("train/mean_reward", mean_reward)
        self.logger.record("train/mean_length", mean_length)
        self.logger.record("train/mean_reward_last_20", mean_reward_last_20)

        self.logger.dump(self.num_timesteps)

    def on_rollout_end(self) -> None:
        self._on_rollout_end()
