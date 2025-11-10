#!/home/plague/pyenv_ros/bin/python3

from CPPO_SITL.drone_env import DroneEnv
from stable_baselines3 import PPO

best_model = f'/home/plague/models/ppoc_quadcopter_final'

model_ppo = PPO.load(best_model)

env = DroneEnv(is_training=False)

obs, info = env.reset()
state_logs = []
action_logs = []

for _ in range(1000):
    action, _states = model_ppo.predict(obs)

    obs, reward, done, truncated, info = env.step(action)
    state_logs.append(info['original_observation'])
    action_logs.append(info['original_action'])

    if(done or truncated):
        env.reset()

env.close()
