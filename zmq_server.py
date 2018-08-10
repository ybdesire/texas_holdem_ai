import zmq
import ast
from RL_brain import DeepQNetwork
import numpy as np


# model
RL = DeepQNetwork(n_actions=7,
                  n_features=29,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

_observation = []
observation = []
reward = 0

# server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind('tcp://127.0.0.1:5555')
while True:
    msg = socket.recv()
    msg = ast.literal_eval(msg.decode('ascii'))
    if('observation' in msg):
        print('get observation')
        print(msg)
        observation = np.array(msg['observation'])
        action = RL.choose_action(observation)
        print('action={0}'.format(action))
        socket.send_string(str(action))
    elif('reward' in msg):
        print('get reward')
        reward = msg['reward']
        observation_ = msg['observation_']
        RL.store_transition(np.array(observation), action, reward, np.array(observation_))
        socket.send_string('completed')
    elif('training' in msg):
        print('need train dqn')
        if hasattr(RL, 'memory_counter'):
            RL.learn()
            print('completed train dqn')
        socket.send_string('completed')
    else:
        print('get sth')
    
    '''
    if msg == 'test':
        socket.send_string('call')
    else:
        socket.send_string('fold')
    '''