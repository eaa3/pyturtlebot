import pybullet as p
import numpy as np

class TurtleBot(object):

    def __init__(self):


        self.turtle = p.loadURDF("data/turtlebot.urdf",[0,0,0])
        self.plane = p.loadURDF("data/plane.urdf")
        self.target_pos = np.zeros(3)

        self.reset()

    def reset(self):
        self.leftWheelVelocity = 0
        self.rightWheelVelocity = 0

    def set_target(self, pos):

        self.target_pos = pos

    def act(self, action):

        forward, turn, speed = action
        # print "action: ", action
        self.rightWheelVelocity = (forward+turn)*speed
        self.leftWheelVelocity = (forward-turn)*speed
        
        p.setJointMotorControl2(self.turtle,0,p.VELOCITY_CONTROL,targetVelocity=self.leftWheelVelocity,force=500)
        p.setJointMotorControl2(self.turtle,1,p.VELOCITY_CONTROL,targetVelocity=self.rightWheelVelocity,force=500)

    def state(self):

        position, orientation = p.getBasePositionAndOrientation(self.turtle)

        rpy = np.array(p.getEulerFromQuaternion(orientation))


        linear, angular = p.getBaseVelocity(self.turtle)
        
        reset = False
        # print "Position: ", position
        if np.linalg.norm(position) > 5.0:
            reset = True

        position = np.array(position)[:2]
        orientation = np.array(orientation)

        direction = (self.target_pos[:2] - position)
        


        norm_direction = np.linalg.norm(direction)

        direction = direction if np.isclose(norm_direction,0.0) else direction/norm_direction
        bearing = np.array([np.cos(rpy[2]), np.sin(rpy[2])])
        berr = direction - bearing#-np.dot(direction,bearing)

        linear = np.array(linear)
        angular = np.array(angular)
        dist = np.linalg.norm(self.target_pos[:2] - position)
        reward = -dist*2.0 #- np.linalg.norm(berr)#- np.linalg.norm(linear)*0.1 - np.linalg.norm(angular)*0.1

        if dist <= 0.01:
            reset = True

        state = np.hstack([position, berr, direction])

        # p0 = np.array([position[0],position[1],0.5])
        # p1 = p0 + np.array([bearing[0],bearing[1],0.0])
        # p2 = p0 + np.array([direction[0],direction[1],0.0])

        

        # p.addUserDebugLine(p0,p1, [1.0,0,0],5, 0.1)
        # p.addUserDebugLine(p0,p2, [1.0,1.0,1.0],5, 0.1)
        # p.addUserDebugLine(p1,p2, [0.0,1.0,0.0],10, 0.1)

        # p3 = np.array([position[0],position[1],0.1])
        # p4 = np.array([self.target_pos[0], self.target_pos[1], 0.1])

        # p.addUserDebugLine(p3,p4, [1.0,0,1.0],10, 0.1)

        
        return state, reward, reset