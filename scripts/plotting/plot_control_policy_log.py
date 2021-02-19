import numpy as np
import argparse
import matplotlib.pyplot as plt
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

output_dir = osp.dirname(args.filename)
# Open npz file
data = np.load(args.filename)

# Get arrays from npz file
step_count        = data["step_count"]
timestamp         = data["timestamp"]
desired_ft_pos    = data["desired_ft_pos"]
desired_ft_vel    = data["desired_ft_vel"]
actual_ft_pos     = data["actual_ft_pos"]
desired_obj_pose  = data["desired_obj_pose"]
desired_obj_vel  = data["desired_obj_vel"]
observed_obj_pose = data["observed_obj_pose"]
observed_filt_obj_pose = data["observed_filt_obj_pose"]
observed_obj_vel = data["observed_obj_vel"]
desired_ft_force  = data["desired_ft_force"]
desired_torque    = data["desired_torque"]

dquat             = data["dquat"]
desired_obj_w     = data["desired_obj_w"]

# Plot desired object trajectory
steps = step_count.shape[0]

plt.figure(figsize=(20,10))
plt.suptitle("Fingertip positions")
plt.subplots_adjust(hspace=0.3)
for f_i in range(3):
    for d_i, dim in enumerate(["x","y","z"]):
        plt.subplot(3,3,f_i*3+d_i+1)
        plt.title("Finger {} dimension {}".format(f_i, dim))
        plt.plot(range(steps), desired_ft_pos[:,f_i*3+d_i], '.', label="Desired")
        plt.plot(range(steps), actual_ft_pos[:,f_i*3+d_i], '.', label="Observed")
        plt.legend()
plt.savefig("{}/ft_pos.png".format(output_dir))

plt.figure(figsize=(10,10))
plt.suptitle("Object pose")
plt.subplots_adjust(hspace=0.3)
for d_i, dim in enumerate(["x","y","z", "qx", "qy", "qz", "qw"]):
    plt.subplot(4,2,d_i+1)
    plt.title("Dimension {}".format(dim))
    plt.plot(range(steps), desired_obj_pose[:,d_i], '.', label="Desired")
    plt.plot(range(steps), observed_obj_pose[:,d_i], '.', label="Observed")
    plt.plot(range(steps), observed_filt_obj_pose[:,d_i], '.', label="Filtered")
    plt.legend()
plt.savefig("{}/obj_pos.png".format(output_dir))

plt.figure(figsize=(10,10))
plt.suptitle("Object velocity")
plt.subplots_adjust(hspace=0.3)
for d_i, dim in enumerate(["x","y","z", "theta_x", "theta_y", "theta_z"]):
    plt.subplot(3,2,d_i+1)
    plt.title("Dimension {}".format(dim))
    plt.plot(range(steps), desired_obj_vel[:,d_i], '.', label="Desired")
    plt.plot(range(steps), observed_obj_vel[:,d_i], '.', label="Observed")
    plt.legend()
plt.savefig("{}/obj_vel.png".format(output_dir))

# Fingertip forces
plt.figure(figsize=(20,10))
plt.suptitle("Desired fingertip forces - world frame")
plt.subplots_adjust(hspace=0.3)
for f_i in range(3):
    for d_i, dim in enumerate(["x","y","z"]):
        plt.subplot(3,3,f_i*3+d_i+1)
        plt.title("Finger {} dimension {}".format(f_i, dim))
        plt.plot(range(steps), desired_ft_force[:,f_i*3+d_i], '.')
plt.savefig("{}/ft_force.png".format(output_dir))

# DEBUGGING PLOTS

# plt.figure(figsize=(10,10))
# plt.suptitle("dquat")
# plt.subplots_adjust(hspace=0.3)
# for d_i, dim in enumerate(["dqx", "dqy", "dqz", "dqw"]):
#     plt.subplot(2,2,d_i+1)
#     plt.title("Dimension {}".format(dim))
#     plt.plot(range(steps-1), dquat[:,d_i], '.')
# plt.savefig("{}/dquat.png".format(output_dir))
#
# plt.figure(figsize=(10,10))
# plt.suptitle("Desired object wrench")
# plt.subplots_adjust(hspace=0.3)
# for d_i, dim in enumerate(["fx", "fy", "fz", "mx", "my", "mz"]):
#     plt.subplot(2,3,d_i+1)
#     plt.title("Dimension {}".format(dim))
#     plt.plot(range(desired_obj_w.shape[0]), desired_obj_w[:,d_i], '.')
# plt.savefig("{}/desired_obj_w.png".format(output_dir))
