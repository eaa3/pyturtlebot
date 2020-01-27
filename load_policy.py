from keras.models import load_model
import numpy as np

import gym
import gym_turtlebot

class LearntPolicy(object):
    
    
    def __init__(self, filename):
        
        self.model = load_model(filename)
        
        
    def act(self, state):
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    


env = gym.make('turtlebot-v0')
state = env.reset()
# observation_space = 6
# action_space = env.action_space.n

agent = LearntPolicy("model.h5")

i = 0
while True:
    
    print("State: ", state, " Shape: ", state.shape)
    state = np.array(state).reshape(1,6)
    env.render()
    action = agent.act(state)
    state_next, reward, terminal, info = env.step(action)
    
    state = state_next
    
    
    
    i += 1
    
    # if i%1000 == 0:
    #     state = env.reset_target()
        
    if terminal:
        state = env.reset()