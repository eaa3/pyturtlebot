import pybullet as p
import time
import numpy as np
import math


### Twist to left and right wheel commands (differential drive kinematics):
### Twist = [vx, vy, vz, wx, wy, wz]
### WheelCMD = [vx - (BASE_WIDTH/2) * wz, vx + (BASE_WIDTH/2) * wz ]

enable_open_gl_rendering = True

def get_image(cam_pos, forward_vector, up_vector):
    width = 320
    height = 240
    fov = 90
    aspect = width / height
    near = 0.02
    far = 5

    # camera pos, look at, camera up direction
    # from_pos = cam_pos+np.array([0.0, 1.0, 0.5])
    # view_matrix = p.computeViewMatrix(from_pos, cam_pos, [0, 0.5, 1])
    cam_pos = cam_pos + np.array([0.0, 0.0, 0.5])
    target_pos = cam_pos + forward_vector*0.1

    view_matrix = p.computeViewMatrix(cam_pos, target_pos, up_vector)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    if enable_open_gl_rendering:
        w, h, rgb, depth, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    else:
        w, h, rgb, depth, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True, renderer=p.ER_TINY_RENDERER)

    depth_buffer = np.reshape(depth, [width, height])
    depth = far * near / (far - (far - near) * depth_buffer)
    seg = np.reshape(seg,[width,height])*1./255.
    return rgb


p.connect(p.GUI)
offset = [0,0,0]

turtle = p.loadURDF("data/turtlebot.urdf",offset)
plane = p.loadURDF("data/plane.urdf")
cube = p.loadURDF("data/marble_cube.urdf", [-1,0,1])
p.setRealTimeSimulation(1)

for j in range (p.getNumJoints(turtle)):
    print(p.getJointInfo(turtle,j))
forward=0
turn=0


#### Simulated LIDAR

numRays = 25
rayFrom=[None]*numRays
rayTo=[None]*numRays
rayForward=[]
rayIds=[None]*numRays
angles=[]
replaceLines=True
        
rayLen = 1.0


rayHitColor = [0,0,1]
rayMissColor = [0,1,0]
rayDefaultColor = [1,0,0]

replaceLines = True

index = 0
totalRays = 0


for i in range (numRays):

        angle = 2.*math.pi*float(i)/numRays
        angles.append(angle)
        ray_forward = np.array([rayLen*math.sin(angle), rayLen*math.cos(angle),0])
        rayForward.append(ray_forward)

while (1):
    p.setGravity(0,0,-10)
    time.sleep(1./240.)
    keys = p.getKeyboardEvents()
    leftWheelVelocity=0
    rightWheelVelocity=0
    speed=10

    position, orientation = p.getBasePositionAndOrientation(turtle)
    orientation_mat = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
    cam_pos = position
    up_vector = orientation_mat[:,2]
    foward_vector = orientation_mat[:,0]
    rgb = get_image(cam_pos, foward_vector, up_vector)


    p.removeAllUserDebugItems()

    position += np.array([0,0,0.5])
    for i in range (numRays):
        rayFrom[i] = position 
        ray_forward = rayForward[i]
        rayTo[i] = position + np.dot(orientation_mat,ray_forward)
        # if (replaceLines):
        #         rayIds[i] = p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
        # else:
        #         rayIds[i] = -1

    
    results = p.rayTestBatch(rayFrom,rayTo)


        
    for i in range (numRays):

        #print(scan[i])
        hitObjectUid=results[i][0]
        angle = angles[i]

        length = results[i][2] #scan[i][2]/1000. #in meters
        fromPosition = position #+ np.dot(orientation_mat,np.array([(length)*math.sin(angleRad),(length)*math.cos(angleRad),0]))
        hitPosition = results[i][3]#np.dot(orientation_mat(np.array([length*math.sin(angleRad),length*math.cos(angleRad),0.1]))
        if hitObjectUid<0:
            p.addUserDebugLine(fromPosition, rayTo[i], rayDefaultColor)#,replaceItemUniqueId=rayIds[index])
        else:
            p.addUserDebugLine(fromPosition,hitPosition, rayHitColor)

    for k,v in keys.items():

        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
            turn = -0.5
        if (k == p.B3G_RIGHT_ARROW and (v&p.KEY_WAS_RELEASED)):
            turn = 0
        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                turn = 0.5
        if (k == p.B3G_LEFT_ARROW and (v&p.KEY_WAS_RELEASED)):
                turn = 0

        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                forward=1
        if (k == p.B3G_UP_ARROW and (v&p.KEY_WAS_RELEASED)):
                forward=0
        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_TRIGGERED)):
                forward=-1
        if (k == p.B3G_DOWN_ARROW and (v&p.KEY_WAS_RELEASED)):
                forward=0

    rightWheelVelocity+= (forward+turn)*speed
    leftWheelVelocity += (forward-turn)*speed
    
    p.setJointMotorControl2(turtle,0,p.VELOCITY_CONTROL,targetVelocity=leftWheelVelocity,force=1000)
    p.setJointMotorControl2(turtle,1,p.VELOCITY_CONTROL,targetVelocity=rightWheelVelocity,force=1000)


