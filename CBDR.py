# CONSTANT BEARING DECREASING RANGE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math
import numpy as np

from vector3 import *

class Missile:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel

def rotate_point_around_axis(point, axis, angle_radians):
    axis = axis / np.linalg.norm(axis)

    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    rotation_matrix = (
        cos_theta * np.eye(3) +
        (1 - cos_theta) * np.outer(axis, axis) +
        sin_theta * np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
    )

    rotated_point = np.dot(rotation_matrix, point)

    return rotated_point

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    t_pos0 = vec3(1000, 0, -3000)
    t_vel0 = vec3(-300, 0, 0)
    t = Missile(t_pos0, t_vel0)

    m_pos0 = vec3(0, 0, 0)
    m_vel0 = vec3(0, 0, -500)
    m = Missile(m_pos0, m_vel0)

    bearing_angle = -float(input("Constant bearing (degrees, 0-90): "))
    bearing_rad = np.deg2rad(bearing_angle)

    m_poses_x = []
    m_poses_y = []
    m_poses_z = []
    t_poses_x = []
    t_poses_y = []
    t_poses_z = []
    dirpoint_x = []
    dirpoint_y = []
    dirpoint_z = []
    times = []
    cycle = 0
    time = 0
    dt = 0.02
    terminate = False
    status = "None"

    print("== RUN SIM ==")
    while not terminate:
        # GUIDANCE
        plane_normal_vector = np.array(t.vel.cross(t.pos - m.pos).normalized().tolist())
        tm = np.array((t.pos - m.pos).normalized().tolist())
        m.vel = vec3(lst=(rotate_point_around_axis(tm, plane_normal_vector, bearing_rad) * m.vel.mag()).tolist())

        # TARGET MANEUVER
        t.vel = vec3(t.vel.mag() * math.sin(time * 0.15), t.vel.mag() * math.cos(time * 0.15), 0)

        # EULER INTEGRATION
        t.pos = t.pos + t.vel * dt
        m.pos = m.pos + m.vel * dt
        m.vel.z += 9.81 * dt

        # SIM DATA RECORD
        m_poses_x.append(m.pos.x)
        m_poses_y.append(m.pos.y)
        m_poses_z.append(-m.pos.z)
        t_poses_x.append(t.pos.x)
        t_poses_y.append(t.pos.y)
        t_poses_z.append(-t.pos.z)

        mt_dist = max((m.pos - t.pos).mag(), m.pos.mag() * 0.05)
        dirpoint_x.append(m.pos.x + m.vel.x / m.vel.mag() * mt_dist)
        dirpoint_y.append(m.pos.y + m.vel.y / m.vel.mag() * mt_dist)
        dirpoint_z.append(-m.pos.z - m.vel.z / m.vel.mag() * mt_dist)

        time = cycle * dt
        times.append(time)

        cycle += 1

        # SIM TERMINATION
        if (m.pos - t.pos).mag() < 30:
            terminate = True
            status = "Hit Target"

        if cycle > 1e5:
            terminate = True
            status = "Timeout"

    print("== END SIM ==")
    print("Status:", status)
    print("Intercept time:", time)

    return times, m_poses_x, m_poses_y, m_poses_z,\
           t_poses_x, t_poses_y, t_poses_z,\
           dirpoint_x, dirpoint_y, dirpoint_z

def plot_traj(ts, mx, my, mz, tx, ty, tz):
    fig = plt.figure()
    ax = plt.figure().add_subplot(projection='3d')
    traj_m, = ax.plot(mx, my, mz)
    traj_t, = ax.plot(tx, ty, tz)

    def update(frame):
        traj_m.set_data(mx[:frame], my[:frame])
        traj_m.set_3d_properties(mz[:frame])

        traj_t.set_data(tx[:frame], ty[:frame])
        traj_t.set_3d_properties(tz[:frame])

    num_frames = len(ts)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=3, repeat=False)
    set_axes_equal(ax)
    plt.show()

def plot_traj(mx, my, mz, tx, ty, tz, dx, dy, dz):
    x1 = mx
    y1 = my
    z1 = mz

    x2 = tx
    y2 = ty
    z2 = tz

    limit_px = max(max(mx), max(tx)) + 1000
    limit_mx = min(min(mx), min(tx)) - 1000
    
    limit_py = max(max(my), max(ty)) + 1000
    limit_my = min(min(my), min(ty)) - 1000

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('-Z (m)')

    line1, = ax.plot(x1, y1, z1, label='Interceptor')
    line2, = ax.plot(x2, y2, z2, label='Target')

    mpoint, = ax.plot([], [], [], marker='o', markersize=2, color='blue')
    tpoint, = ax.plot([], [], [], marker='o', markersize=2, color='orange')

    line3, = ax.plot([], [], [], label='Missile Heading', linestyle="dashed")
    line4, = ax.plot([], [], [], label='Constant Bearing', linestyle="dashed")

    terrainX = np.arange(limit_mx, limit_px, 500)
    terrainY = np.arange(limit_my, limit_py, 500)
    terrainX, terrainY = np.meshgrid(terrainX, terrainY)
    terrainZ = np.sin(0 * terrainX)
    ax.plot_surface(terrainX, terrainY, terrainZ, color="bisque")

    def update(frame):
        line1.set_data(x1[:frame], y1[:frame])
        line1.set_3d_properties(z1[:frame])

        line2.set_data(x2[:frame], y2[:frame])
        line2.set_3d_properties(z2[:frame])

        mpoint.set_data([x1[frame]], [y1[frame]])
        mpoint.set_3d_properties([z1[frame]])

        tpoint.set_data([x2[frame]], [y2[frame]])
        tpoint.set_3d_properties([z2[frame]])

        line3.set_data([x1[frame], dx[frame]], [y1[frame], dy[frame]])
        line3.set_3d_properties([z1[frame], dz[frame]])

        line4.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        line4.set_3d_properties([z1[frame], z2[frame]])

    num_frames = len(x1)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=0.02, repeat=False)

    set_axes_equal(ax)
    plt.legend()
    plt.show()

ts, mx, my, mz, tx, ty, tz, dx, dy, dz = main()
plot_traj(mx, my, mz, tx, ty, tz, dx, dy, dz)
