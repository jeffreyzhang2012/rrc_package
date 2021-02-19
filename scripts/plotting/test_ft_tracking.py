import numpy as np
import pandas
import argparse
import matplotlib.pyplot as plt

import rrc_iprl_package.traj_opt.kinematics_utils as k_utils

POINT_SKIP = 100
STEPS_TO_PLOT = 30000

"""
Test fingertip position tracking
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("stdout", type=str)
    parser.add_argument("fig_file", type=str)
    args = parser.parse_args()

    data = pandas.read_csv(args.filename, delim_whitespace=True, header=0, low_memory=False)
    # Logs from user code
    stdout = pandas.read_csv(args.stdout, delim_whitespace=False, header=0, low_memory=False)

    # Get actual joint positions from data
    observation_times = []
    observation_ft_pos = []
    observation_start_time = data[["timestamp"]].iloc[0]
    stdout_times = []
    stdout_start_time = stdout[["timestamp"]].iloc[0]

    startup_time = stdout_start_time - observation_start_time
    print("observation start time: {}".format(observation_start_time))
    print("stdout start time: {}".format(stdout_start_time - observation_start_time))
    # Find index in data with timestamp closest to stdout_start_time
    # Sort by timestamp value differences to stdout_start_time
    user_code_start_idx = int(data.iloc[(data["timestamp"]-stdout_start_time["timestamp"]).abs().argsort()[:1]]["#time_index"].tolist()[0])

    user_code_start_idx = 0

    plot_range = [user_code_start_idx, user_code_start_idx + STEPS_TO_PLOT*4]

    for index, row in data.iterrows():
        if index % POINT_SKIP != 0: continue
        if index <= plot_range[0] or index >= plot_range[1]: continue
        print(index)
        observation_times.append(row["timestamp"] - observation_start_time)
        cur_q = [
                row["observation_position_0"],
                row["observation_position_1"],
                row["observation_position_2"],
                row["observation_position_3"],
                row["observation_position_4"],
                row["observation_position_5"],
                row["observation_position_6"],
                row["observation_position_7"],
                row["observation_position_8"],
                ]
        cur_ft_pos = k_utils.FK(cur_q)
        observation_ft_pos.append(cur_ft_pos)

    # Goal fingertip positions...? Need to be parsed from user_stdout.txt
    observation_ft_pos = np.array(observation_ft_pos)

    # Plot
    plt.figure(figsize=(20,20))
    plt.suptitle("Fingertip positions")
    plt.subplots_adjust(hspace=0.3)
    for f_i in range(3):
        for d_i, dim in enumerate(["x","y","z"]):
            plt.subplot(3,3,f_i*3+d_i+1)
            plt.title("Finger {} dimension {}".format(f_i, dim))
            plt.scatter(observation_times, observation_ft_pos[:,f_i*3+d_i],label="Observed")
            plt.scatter(stdout[["timestamp"]].iloc[0:STEPS_TO_PLOT] - stdout_start_time + startup_time, stdout[["desired_ft_pos_{}".format(f_i*3+d_i)]].iloc[0:STEPS_TO_PLOT].to_numpy(), label="Desired")
            plt.legend()
    plt.savefig(args.fig_file)


if __name__ == "__main__":
    main()
