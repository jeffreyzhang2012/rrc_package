import numpy as np

import pinocchio

from trifinger_simulation.pinocchio_utils import Kinematics

class CustomPinocchioUtils(Kinematics):
    """
    Consists of kinematic methods for the finger platform.
    """
    m1=0.2
    m2=0.2
    m3=0.01
    # m1=0.2
    # m2=0.2
    # m3=0.01
    ms=[m1,m2,m3]
    I1=np.zeros((3,3))
    # np.fill_diagonal(I1,[3.533e-4,5.333e-5,3.533e-4])
    np.fill_diagonal(I1,[4.59-4,6.93e-5,4.59e-4])
    I2=np.zeros((3,3))
    # np.fill_diagonal(I2,[3.533e-4,3.533e-4,5.333e-5])
    np.fill_diagonal(I2,[4.41e-4,4.41e-4,6.67e-5])
    I3=np.zeros((3,3))
    # np.fill_diagonal(I3,[1.667e-5,1.667e-5,6.667e-7])
    np.fill_diagonal(I3,[3.5e-5,3.5e-5,1.4e-6])
    Is=[I1,I2,I3]

    def __init__(self, finger_urdf_path, tip_link_names):
        """
        Initializes the finger model on which control's to be performed.
    
        Args:
            finger (SimFinger): An instance of the SimFinger class
        """
        super().__init__(finger_urdf_path, tip_link_names)
    
    def get_tip_link_jacobian(self, finger_id, q):
        """
        Get Jacobian for tip link of specified finger
        All other columns are 0
        """
        pinocchio.computeJointJacobians(
            self.robot_model, self.data, q,
        )
        #pinocchio.framesKinematics(
        #    self.robot_model, self.data, q,
        #)
        pinocchio.framesForwardKinematics(
            self.robot_model, self.data, q,
        )
        frame_id = self.tip_link_ids[finger_id]
        Ji = pinocchio.getFrameJacobian(
          self.robot_model,
          self.data,
          frame_id,
          pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
    
        #print(self.robot_model.frames[frame_id].placement)
        #print(self.data.oMf[frame_id].rotation)
        return Ji

    def get_any_link_jacobian(self, frame_id, q):
        """
        Get Jacobian for tip link of specified finger
        All other columns are 0
        """
        pinocchio.computeJointJacobians(
            self.robot_model, self.data, q,
        )
        pinocchio.framesForwardKinematics(
            self.robot_model, self.data, q,
        )
        Ji = pinocchio.getFrameJacobian(
          self.robot_model,
          self.data,
          frame_id,
          pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return Ji #6x9
    
    def get_lambda_and_g_matrix(self, finger_id, q, Jvi):
        Ai = np.zeros((9,9))
        g = np.zeros(9)
        grav = np.array([0,0,-9.81])
        order = [0,1,3]
        for j in range(3):
            id = (finger_id+1)*10+order[j]
            Jj = self.get_any_link_jacobian(id, q)
            Jjv = Jj[:3,:]
            Jjw = Jj[3:,:]
            g -= self.ms[j]*Jjv.T @ grav * 0.33
            Ai += (self.ms[j]*Jjv.T @ Jjv + Jjw.T @ self.Is[j] @ Jjw)*0.33;
          # Ai is kinetic energy matrix in configuration space
        Jvi_inv = np.linalg.pinv(Jvi)
        Li = Jvi_inv.T @ Ai @ Jvi_inv
        # Li = Ai;
        # Li is Lambda matrix (kinetic energy matrix in operation space)
        return Li,g
