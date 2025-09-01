import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN

# === Custom driving environment ===
class DrivingEnv(gym.Env):
    def __init__(self):
        super(DrivingEnv, self).__init__()
        self.observation_space = spaces.Box(low=np.array([0,0,-30,0]),
                                            high=np.array([120,1,30,1]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.state = None

    def reset(self):
        self.state = np.array([50, 0.5, 0, 0], dtype=np.float32)
        return self.state

    def step(self, action):
        speed, throttle, steering, brake = self.state
        if action == 0:
            speed = max(0, speed - 5); brake = 1.0
        elif action == 1:
            speed = min(120, speed + 5); throttle = 1.0
        elif action == 2:
            steering = min(30, steering + 5)
        elif action == 3:
            steering = max(-30, steering - 5)

        reward = 0
        if 40 <= speed <= 80: reward += 1
        if abs(steering) > 20: reward -= 1
        if brake > 0.8 and speed > 20: reward -= 1

        self.state = np.array([speed, throttle, steering, brake], dtype=np.float32)
        done = False
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")

# === Train DQN model ===
env = DrivingEnv()
drl_model = DQN("MlpPolicy", env, verbose=1)
drl_model.learn(total_timesteps=5000)

# === Test the trained agent ===
obs = env.reset()
for _ in range(10):
    action, _states = drl_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
