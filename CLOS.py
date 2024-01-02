# COMMAND TO LINE OF SIGHT

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

    K_p = float(input("Enter angular correction factor (around 5 is a good pick): ")) * 0.1

    m_poses_x = []
    m_poses_y = []
    m_poses_z = []
    t_poses_x = []
    t_poses_y = []
    t_poses_z = []
    times = []
    cycle = 0
    time = 0
    dt = 0.02
    terminate = False
    status = "None"

    print("== RUN SIM ==")
    while not terminate:
        # GUIDANCE
        if time > dt:
            m_vel_past = m.vel.mag()
            
            t_dir = t.pos / t.pos.mag()
            m_dir = m.pos / m.pos.mag()
            
            vt_theta = t.vel - t_dir * t.vel.dot(t_dir)
            vm_theta = m.vel - m_dir * m.vel.dot(m_dir)

            lineshift = (t.pos - m.pos) - m_dir * (m_dir.dot((t.pos - m.pos)))

            psi = math.acos(m.pos.dot(t.pos) / (m.pos.mag() * t.pos.mag()))
            
            vm_desired = vt_theta * (m.pos.mag() / t.pos.mag()) + lineshift * K_p

            if (vm_desired.mag() > m.vel.mag()):
                m.vel = vm_desired

            else:
                m.vel = vm_desired + m_dir * math.sqrt(m.vel.mag()**2 - vm_desired.mag()**2)

            m.vel = m.vel / m.vel.mag() * m_vel_past

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

        time = cycle * dt
        times.append(time)

        cycle += 1

        # SIM TERMINATION
        if (m.pos - t.pos).mag() < 30:
            terminate = True
            status = "Hit Target"

        if cycle > 1e3:
            terminate = True
            status = "Timeout"

    print("== END SIM ==")
    print("Status:", status)
    print("Intercept time:", time)

    return times, m_poses_x, m_poses_y, m_poses_z,\
           t_poses_x, t_poses_y, t_poses_z

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

def plot_traj(mx, my, mz, tx, ty, tz):
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
    line3, = ax.plot([0, x1[0]], [0, y1[0]], [0, z1[0]], label='Missile LoS', linestyle="dashed", color='blue')
    line4, = ax.plot([0, x2[0]], [0, y2[0]], [0, z2[0]], label='Target LoS', linestyle="dashed", color='orange')

    mpoint, = ax.plot([], [], [], marker='o', markersize=2, color='blue')
    tpoint, = ax.plot([], [], [], marker='o', markersize=2, color='orange')

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
        
        line3.set_data([0, x1[frame]], [0, y1[frame]])
        line3.set_3d_properties([0, z1[frame]])
        
        line4.set_data([0, x2[frame]], [0, y2[frame]])
        line4.set_3d_properties([0, z2[frame]])

        mpoint.set_data([x1[frame]], [y1[frame]])
        mpoint.set_3d_properties([z1[frame]])

        tpoint.set_data([x2[frame]], [y2[frame]])
        tpoint.set_3d_properties([z2[frame]])

    num_frames = len(x1)
    animation = FuncAnimation(fig, update, frames=num_frames, interval=0.02, repeat=False)

    set_axes_equal(ax)
    plt.legend()
    plt.show()

ts, mx, my, mz, tx, ty, tz = main()
plot_traj(mx, my, mz, tx, ty, tz)
