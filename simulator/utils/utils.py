import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from enum import IntEnum


class AgentType(IntEnum):
    ADULT = 0
    BICYCLE = 1
    CHILD = 2
    ADULT_STATIC = 3
    ROBOT = 4


def plot_value_heatmap(states, global_step, action_values, robot):
    assert robot.kinematics == 'holonomic'
    for agent in [states[global_step][0]] + states[global_step][1]:
        print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                    agent.vx, agent.vy, agent.theta))
    # when any key is pressed draw the action value plot
    fig, axis = plt.subplots()
    speeds = [0] + robot.policy.speeds
    rotations = robot.policy.rotations + [np.pi * 2]
    r, th = np.meshgrid(speeds, rotations)
    z = np.array(action_values[global_step % len(states)][1:])
    z = (z - np.min(z)) / (np.max(z) - np.min(z))
    z = np.reshape(z, (16, 5))
    polar = plt.subplot(projection="polar")
    polar.tick_params(labelsize=16)
    mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
    plt.plot(rotations, r, color='k', ls='none')
    plt.grid()
    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(mesh, cax=cbaxes)
    cbar.ax.tick_params(labelsize=16)
    plt.show()
