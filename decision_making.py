from RL_brain import DeepQNetwork
import numpy as np


RL = DeepQNetwork(n_actions=7,
                  n_features=29,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)




observation=np.array([0]*29)
action = RL.choose_action(observation)
print(action)
'''
observation_, reward, done, info = env.step(action)
RL.store_transition(observation, action, reward, observation_)

if total_steps > 1000:
    RL.learn()
'''









