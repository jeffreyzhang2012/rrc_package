import numpy as np
from sympy import *

"""
Compute forward kinematics and Jacobian analytically
"""

BASE_ANGLE_DEGREES = [0, -120, -240]

# Fixed values from URDF
theta_base = symbols("theta_base")

# joint origin xyz values w.r.t previous joint
j1_xyz = [0, 0, 0] # joint 1
j2_xyz = [0.01685, 0.0505, 0] # joint 2
j3_xyz = [0.04922, 0, -0.16] # joint 3
j4_xyz = [0.0185, 0, -0.1626] # joint 4

# Joint values
q1, q2, q3 = symbols("q1 q2 q3")

H_0_wrt_base = Matrix([
                    [cos(theta_base), -sin(theta_base), 0, 0],
                    [sin(theta_base), cos(theta_base), 0, 0],
                    [0, 0, 1, 0.29],
                    [0, 0, 0, 1],
                    ])

# Frame 1 w.r.t. frame 0
# Rotation around y axis
H_1_wrt_0 = Matrix([
                    [cos(q1),       0, sin(q1), j1_xyz[0]],
                    [0,             1,       0, j1_xyz[1]],
                    [-sin(q1),      0, cos(q1), j1_xyz[2]],
                    [0, 0, 0, 1],
                    ])

# Frame 2 w.r.t. frame 1
# Rotation around x axis
H_2_wrt_1 = Matrix([
                  [1,       0,        0, j2_xyz[0]],
                  [0, cos(q2), -sin(q2), j2_xyz[1]],
                  [0, sin(q2),  cos(q2), j2_xyz[2]],
                  [0,       0,        0,         1],
                  ])

# Frame 3 w.r.t. frame 2
# Rotation around x axis
H_3_wrt_2 = Matrix([
                  [1,       0,        0, j3_xyz[0]],
                  [0, cos(q3), -sin(q3), j3_xyz[1]],
                  [0, sin(q3),  cos(q3), j3_xyz[2]],
                  [0,       0,        0,         1],
                  ])

# Transformation from frame 3 to 4
# Fixed
H_4_wrt_3 = Matrix([
                  [1, 0, 0, j4_xyz[0]],
                  [0, 1, 0, j4_xyz[1]],
                  [0, 0, 1, j4_xyz[2]],
                  [0, 0, 0,         1],
                  ])

H_4_wrt_0 = H_0_wrt_base @ H_1_wrt_0 @ H_2_wrt_1 @ H_3_wrt_2 @ H_4_wrt_3

# Reference frame 5 attached to very end of fingertip
# Transformation from frame 4 to 5
# Fixed
H_5_wrt_4 = Matrix([
                  [1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, -0.0095], # TODO HARDCODED
                  [0, 0, 0, 1],
                  ])

H_5_wrt_0 = H_4_wrt_0 @ H_5_wrt_4

# Compose rotations to get orientation of frame 4 w.r.t frame 0
R_0_wrt_base = H_0_wrt_base[:3,:3]
R_1_wrt_0 = H_1_wrt_0[:3,:3]
R_2_wrt_1 = H_2_wrt_1[:3,:3]
R_3_wrt_2 = H_3_wrt_2[:3,:3]
R_4_wrt_3 = H_4_wrt_3[:3,:3]

R_4_wrt_0 = R_0_wrt_base @ R_1_wrt_0 @ R_2_wrt_1 @ R_3_wrt_2 @ R_4_wrt_3

print("{}".format(R_4_wrt_0).replace("cos", "np.cos").replace("sin", "np.sin").replace("theta_base", "theta"))

p = np.array([[0],[0],[0],[1]])

"""
Compute forward kinematics for all fingers given joint positions q
"""
def FK(q):
    ft_pos = []
    for fplit
i, angle in enumerate(BASE_ANGLE_DEGREES):
        theta = angle * (np.pi/180)
        q1_val = q[3*f_i + 0]
        q2_val = q[3*f_i + 1]
        q3_val = q[3*f_i + 2]
        eef_wf = H_4_wrt_0 @ p
        eef_wf = eef_wf[0:3, :]
        p_wf = eef_wf.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta})
        ft_pos += flatten(p_wf)
    return ft_pos

def get_H_5_wrt_0(q):
    H_list = []
    for f_i, angle in enumerate(BASE_ANGLE_DEGREES):
        theta = angle * (np.pi/180)
        q1_val = q[3*f_i + 0]
        q2_val = q[3*f_i + 1]
        q3_val = q[3*f_i + 2]
        H = H_5_wrt_0.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta})
        H_list.append(H)
    return H_list

"""
Get fingertip reference frame orientation for all fingers
"""
def get_ft_R_sympy(q):
    R_list = []
    for f_i, angle in enumerate(BASE_ANGLE_DEGREES):
        theta = angle * (np.pi/180)
        q1_val = q[3*f_i + 0]
        q2_val = q[3*f_i + 1]
        q3_val = q[3*f_i + 2]
        R = R_4_wrt_0.subs({"q1": q1_val, "q2": q2_val, "q3": q3_val, "theta_base": theta})
        R_list.append(R)
    return R_list

#dq1 = eef_wf.diff(q1)
#dq2 = eef_wf.diff(q2)
#dq3 = eef_wf.diff(q3)
#
## Compute jacobian
#J = dq1.row_join(dq2).row_join(dq3)
##print("sympy J")
##print(J)
#
#############################################
## Test forward kinematics , and Jacobian for one finger
#############################################
#f_id = 0
#theta_base_deg = 0 # Fixed andle of finger base w.r.t. center holder {0, 120, 240} degrees
#theta_base_val = theta_base_deg * (np.pi/180)
#
#q1_val = 0
#q2_val = 0.7
#q3_val = -1.7
#
#q_all = np.array([q1_val, q2_val, q3_val,q1_val, q2_val, q3_val,q1_val, q2_val, q3_val])
#
#
#print("eef_wf analytical: {}".format(p_wf))

