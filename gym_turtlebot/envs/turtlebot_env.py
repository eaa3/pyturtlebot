import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
from gym_turtlebot.envs.turtlebot import TurtleBot
import numpy as np

class TurtleBotEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  ACTION = [(5.0, 0.0, 5.0), (-5.0, 0.0, 5.0), 
            (0.0, 0.25, 5.0), (0.0, -0.25, 5.0),
            (0.0, 0.0, 0.0)] # no op

  def __init__(self):
    p.connect(p.GUI)
    p.setGravity(0,0,-10)
    p.setRealTimeSimulation(0)
    p.setTimeStep(0.025)
    self.robot = TurtleBot()

    self.action_space = spaces.Discrete(4)

    self.observation_space = spaces.Box(low=-4, high=4,
                                            shape=(6,))

    self.box = p.loadURDF("data/marble_cube.urdf", [-1,0,1],globalScaling=0.5)

  def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = TurtleBotEnv.ACTION[action]

        self.robot.act(action_type)

  def step(self, action):

    self._take_action(action)

    p.stepSimulation()

    observation, reward, reset = self.robot.state()

    # print observation
    return observation, reward, reset, {}
    # observation, reward, done, info

  def reset(self):
    # p.resetSimulation()
    # self.robot.act([0,0,0])
    p.resetBasePositionAndOrientation(self.robot.turtle, [0,0,0], [0,0,0,0.05])
    p.resetBaseVelocity(self.robot.turtle, [0,0,0],[0,0,0])

    position = np.ones(3)*0.5 + np.ones(3)*4 - np.random.rand(3)*8#np.ones(3)*1.0 #+ np.random.rand(3) 
    position[2] = 0.3
    p.resetBasePositionAndOrientation(self.box, position, [0,0,0,0.05])

    self.robot.set_target(position)

    state, _, _ = self.robot.state()
    return state

  def reset_target(self):
    position = np.ones(3)*0.5 + np.ones(3)*4 - np.random.rand(3)*8#np.ones(3)*1.0 #+ np.random.rand(3) 
    position[2] = 0.3
    p.resetBasePositionAndOrientation(self.box, position, [0,0,0,0.05])

    self.robot.set_target(position)

    state, _, _ = self.robot.state()




    return state

  def render(self, mode='human'):
    return  np.zeros(6)

  def close(self):
    pass