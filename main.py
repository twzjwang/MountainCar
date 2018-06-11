import numpy as np
import random
import pandas as pd
import gym

def toPos(x, y):
    return int((y + 0.1) / 0.2 * 100) * 100 + int((x + 1.5) / 2.5 * 100)

env = gym.make('MountainCar-v0')
Q = np.zeros([100 * 100, env.action_space.n])

record = np.zeros(100)
for i in range(30000):
    observation = env.reset()
    pos = toPos(observation[0], observation[1])
    reward_sum = 0.0
    for t in range(300):
        #env.render()
        action = np.argmax(Q[pos,:]+random.randint(0,2)*(1.0/(i+1)))
        observation, new_reward, done, info = env.step(action)
        new_pos = toPos(observation[0], observation[1])
        Q[pos, action] = new_reward + Q[pos, action] + np.max(Q[new_pos, :]) - Q[pos, action]
        reward_sum += new_reward
        pos = new_pos
        if i % 5000 == 0:
            env.render()
        if done:
            break
    if i % 100 == 0:
        mean = record.mean()
        print(i, mean)
    else:
        record[i % 100] = reward_sum
