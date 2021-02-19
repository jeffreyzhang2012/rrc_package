import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()
dest_file = args.filename

save_path = os.path.splitext(dest_file)[0]

with open(dest_file,'r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ' ',
                           quotechar = '"')
    data = [data for data in data_iter]
data_array = np.asarray(data)
names = data_array[0,:]
data_array = data_array[1:,:].astype(np.float)
##
dof = 9;
tstart = 2000
tn = 5000
step = 100
time = data_array[1+tstart:tn+tstart:step,1]
time = time - time[0]

##torque
idx = np.where(names=='observation_torque_0')[0][0]
obs_torque = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
idx = np.where(names=='applied_action_torque_0')[0][0]
app_torque = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
idx = np.where(names=='desired_action_torque_0')[0][0]
des_torque = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
torque,torque_ax = plt.subplots(3,3,figsize=(30,20))
torque.suptitle('torque')
for i in range(dof):
    torque_ax[i//3][i%3].plot(time,obs_torque[:,i],label='observed');
    torque_ax[i//3][i%3].plot(time,app_torque[:,i],label='applied');
    torque_ax[i//3][i%3].plot(time,des_torque[:,i],label='desired');
    torque_ax[i//3][i%3].legend()
    torque_ax[i//3][i%3].title.set_text('Joint %d'%i)

plt.savefig("{}_torque.png".format(save_path))

##position
idx = np.where(names=='observation_position_0')[0][0]
obs_position = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
idx = np.where(names=='applied_action_position_0')[0][0]
app_position = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
idx = np.where(names=='desired_action_position_0')[0][0]
des_position = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
position,position_ax = plt.subplots(3,3,figsize=(30,20))
position.suptitle('position')
for i in range(dof):
    position_ax[i//3][i%3].plot(time,obs_position[:,i],label='observed');
    position_ax[i//3][i%3].plot(time,app_position[:,i],label='applied');
    position_ax[i//3][i%3].plot(time,des_position[:,i],label='desired');
    position_ax[i//3][i%3].legend()
    position_ax[i//3][i%3].title.set_text('Joint %d'%i)
plt.savefig("{}_position.png".format(save_path))
##velocity
idx = np.where(names=='observation_velocity_0')[0][0]
obs_velocity = (data_array[1+tstart:tn+tstart:step,idx:idx+dof])
velocity,velocity_ax = plt.subplots(3,3,figsize=(30,20))
velocity.suptitle('velocity')
for i in range(dof):
    velocity_ax[i//3][i%3].plot(time,obs_velocity[:,i],label='observed');
    velocity_ax[i//3][i%3].legend()
    velocity_ax[i//3][i%3].title.set_text('Joint %d'%i)

plt.savefig("{}_velocity.png".format(save_path))
