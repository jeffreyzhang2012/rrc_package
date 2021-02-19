import pybullet
import numpy as np
from trifinger_simulation.tasks import move_cube


MAX_DIST = move_cube._max_cube_com_distance_to_center
#DIST_THRESH = move_cube._CUBE_WIDTH / 5
ORI_THRESH = np.pi / 8
REW_BONUS = 1
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


def reset_camera():
    camera_pos = (0.,0.2,-0.2)
    camera_dist = 1.0
    pitch = -45.
    yaw = 0.
    if pybullet.isConnected() != 0:
        pybullet.resetDebugVisualizerCamera(cameraDistance=camera_dist,
                                    cameraYaw=yaw,
                                    cameraPitch=pitch,
                                    cameraTargetPosition=camera_pos)


