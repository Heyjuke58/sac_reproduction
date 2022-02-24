from TD3.TD3 import TD3
import gym
import numpy as np

env = gym.make("Hopper-v3")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

td3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)

td3.load("models/TD3_Hopper-v3_1337")


observation = env.reset()
for _ in range(10000):
    env.render()
    action = td3.select_action(np.array(observation))
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()