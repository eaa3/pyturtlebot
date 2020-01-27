import gym
import gym_turtlebot
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import random

import matplotlib.pyplot as plt

# env = gym.make('turtlebot-v0')
# env.reset()


# Deep Q-learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.90    # discount rate
        self.epsilon = 1.0#1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        print("Eps: ", self.epsilon)
    def save(self):
        
        self.model.save("model.h5")

def turtle_train():
    env = gym.make('turtlebot-v0')
    state = env.reset()
    observation_space = 6
    action_space = env.action_space.n
    dqn_solver = DQNAgent(observation_space, action_space)
    i = 0
    avg_reward = 0.0
    max_iterations = 500000#100000
    expected_return = None
    
    returns = []
    plt.figure(0)
    plt.show(False)
    while i < max_iterations:
        # state = env.reset()

        if i%10000==0:
            state = env.reset_target()
        state = np.reshape(state, [1, observation_space])
        # for t in range(500):
        env.render()
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)
        avg_reward = avg_reward*0.99 + reward*0.01
        # print("Reward: ", avg_reward)
        # reward = reward if not terminal else -reward
        state_next = np.reshape(state_next, [1, observation_space])
        dqn_solver.remember(state, action, reward, state_next, terminal)
            
        state = state_next
        if terminal:
            state = env.reset()
            # break

        i+=1
        print("%d/%d"%(i, max_iterations))
        if i > 100 and i%500==0:
            dqn_solver.replay(32)
            
            if expected_return is None:
                expected_return = avg_reward
            else:
                expected_return = avg_reward*0.001 + expected_return*0.999
            returns.append(expected_return)

            plt.plot(range(len(returns)), returns)
            plt.draw()
            plt.pause(0.001)
    dqn_solver.save()
    # plt.show()
turtle_train()
# for _ in range(500):
#     env.render()
#     env.step(env.action_space.sample())