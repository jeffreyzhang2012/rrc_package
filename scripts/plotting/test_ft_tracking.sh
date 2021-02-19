#!/bin/sh
# Executes procedure to run sim and plot fingertip tracking data

output_dir=$1

mkdir -p $output_dir

echo "Saving in $output_dir"

#echo "Run code in simulation to collect data"
# Run code in simulation to collect data
#~/rrc_package/run_in_simulation.py --output-dir $output_dir --repository ~/rrc_package --branch old-test --backend-image ~/realrobotchallenge.sif --user-image ~/user_image.sif

#echo "Simulation done. Processing data"

# Convert robot_data.dat to .csv
~/realrobotchallenge.sif rosrun robot_fingers robot_log_dat2csv.py "${output_dir}robot_data.dat" "${output_dir}robot_data.csv"

echo "Plotting data"
# Run script to plot data
python test_ft_tracking.py "${output_dir}robot_data.csv" "${output_dir}user/control_policy_data.csv" "${output_dir}ft_pos.png"

echo "Finished."
