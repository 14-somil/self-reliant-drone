#!/home/plague/pyenv_ros/bin/python3

#TODO:
#1. fix compass calibration issue
#2. fix mesagging
#3. check euler angle calculations

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList, EvalCallback

from CPPO_SITL.drone_env import DroneEnv
from CPPO_SITL.callback import TensorboardCallback
import torch
import os
import sys

register(id='DroneEnv', entry_point=DroneEnv)

env = make_vec_env('DroneEnv', n_envs=1)

eval_env = env

onpolicy_kwargs = dict(
    activation_fn=torch.nn.LeakyReLU,
    net_arch=[dict(pi=[64,128,256], vf=[64,128,256])]
)

ppo_params = {
    "policy":ActorCriticPolicy,  #"MlpPolicy",acpolicy,
    "env": env,
    "learning_rate": 1e-4,#A too low learning rate can result in very slow convergence.
                        #A too high learning rate can cause the training to converge too quickly to a suboptimal solution
    "n_steps": 2048, #The number of environment steps to collect before performing an update.
    "batch_size": 64, #Number of samples used for each gradient update.
    "n_epochs": 20, #Number of passes over the data during each update.

    "gamma": 0.95, #Discount factor for future rewards.A high gamma makes the agent prioritize long-term rewards, while a low gamma focuses on short-term rewards
    "gae_lambda": 0.90, #Smoothing parameter for Generalized Advantage Estimation (GAE).Balances bias and variance in the advantage function estimate.
    "clip_range": 0.2, #Clipping parameter for the probability ratio. Prevents large policy updates that can destabilize training
    "ent_coef": 0.01,#Encourages exploration by adding the entropy of the policy to the loss function
    "vf_coef": 0.5, #Scales the contribution of the value function loss
    "max_grad_norm": 0.5, #
    "verbose": 1,
    "stats_window_size": 200,
    # "policy_kwargs":None,
    "seed": 17,
    "device": "cpu",
    "tensorboard_log":"./tensorboard/ppo_tensorboard",
    "policy_kwargs":onpolicy_kwargs
}

model_ppo = None

if(len(sys.argv) > 1):
    model_ppo = PPO.load(sys.argv[-1], env=env)
else:
    model_ppo = PPO(**ppo_params)

models_dir = "./models/"
os.makedirs(models_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(save_freq=1000000, save_path='./models/', name_prefix='ppo_quadcopter')
tensorboard_callback = TensorboardCallback()
eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model', log_path='./logs/', eval_freq=5000, deterministic=True, render=False)
callback = CallbackList([checkpoint_callback, tensorboard_callback, eval_callback])

model_ppo.learn(total_timesteps=4000000, callback=callback)

model_ppo.save("ppoc_quadcopter_final")
