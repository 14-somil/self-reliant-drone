#!/home/plague/pyenv_ros/bin/python3

from CPPO_SITL.drone_env import DroneEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

def main():
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

    state_logs = np.array(state_logs)
    action_logs = np.array(action_logs)

    state_logs_flat = state_logs

    fig, axs = plt.subplots(4, 1, figsize=(8, 8), dpi=300)

    axs[0].plot(state_logs_flat[:, 0], label='x')
    axs[0].plot(state_logs_flat[:, 1], label='y')
    axs[0].plot(state_logs_flat[:, 2], label='z')
    axs[0].set_title('Position')
    axs[0].legend()

    axs[1].plot(state_logs_flat[:, 3], label='roll')
    axs[1].plot(state_logs_flat[:, 4], label='pitch')
    axs[1].plot(state_logs_flat[:, 5], label='yaw')
    axs[1].set_title('Orientation (Euler angles)')
    axs[1].legend()

    axs[2].plot(state_logs_flat[:, 6], label='vx')
    axs[2].plot(state_logs_flat[:, 7], label='vy')
    axs[2].plot(state_logs_flat[:, 8], label='vz')
    axs[2].set_title('Linear Velocity')
    axs[2].legend()

    axs[3].plot(state_logs_flat[:, 9], label='wx')
    axs[3].plot(state_logs_flat[:, 10], label='wy')
    axs[3].plot(state_logs_flat[:, 11], label='wz')
    axs[3].set_title('Angular Velocity')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig("state_logs_plot.png", dpi=300)

    # Plotting the actions
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    axs[0].plot(action_logs[:, 0], label='thrust_1')
    axs[0].set_title('Thrust Command 1')
    axs[0].legend()

    axs[1].plot(action_logs[:, 1], label='thrust_2')
    axs[1].set_title('Thrust Command 2')
    axs[1].legend()

    axs[2].plot(action_logs[:, 2], label='thrust_3')
    axs[2].set_title('Thrust Command 3')
    axs[2].legend()

    axs[3].plot(action_logs[:, 3], label='thrust_4')
    axs[3].set_title('Thrust Command 4')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig("action_logs_plot.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
