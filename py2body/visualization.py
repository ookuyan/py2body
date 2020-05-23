#!/usr/bin/env python

__all__ = ['plot_orbit', 'animate_orbit']

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, writers

from PIL import Image


# plt.style.use('dark_background')

textures = {
    'sun': './py2body/data/sun.jpg',
    'mercury': './py2body/data/mercury.jpg',
    'venus': './py2body/data/venus_atmosphere.jpg',
    'earth': './py2body/data/earth_day.jpg',
    'moon': './py2body/data/moon.jpg',
    'mars': './py2body/data/mars.jpg',
    'jupiter': './py2body/data/jupiter.jpg',
    'saturn': './py2body/data/saturn.jpg',
    'uranus': './py2body/data/uranus.jpg',
    'neptune': './py2body/data/neptune.jpg',
    'wire_frame': 'wire_frame',
    'surface': 'surface'
}


def create_sphere(ax, radius=1, position=None, texture='wire_frame',
                  texture_alpha=0.8, color='Blues', texture_bin=8,
                  rstride=4, cstride=4, title=None, x_label=None,
                  y_label=None, z_label=None, quiver=True):

    sphere = dict()

    if title:
        ax.set_title(title)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if z_label:
        ax.set_zlabel(z_label)

    if texture not in ['wire_frame', 'surface']:
        try:
            texture = textures[texture]
        except KeyError:
            pass

        img = Image.open(texture)
        img = np.array(
            img.resize([int(d / texture_bin) for d in img.size])) / 256.

        lons = np.linspace(-np.pi, np.pi, img.shape[1])
        lats = np.linspace(-np.pi/2, np.pi/2, img.shape[0])[::-1]

        if position is None:
            position = np.array([0, 0, 0])

        rx, ry, rz = position

        x = rx + radius * np.outer(np.cos(lons), np.cos(lats)).T
        y = ry + radius * np.outer(np.sin(lons), np.cos(lats)).T
        z = rz + radius * np.outer(np.ones(np.size(lons)), np.sin(lats)).T

        sphere['texture'] = img

        ax.plot_surface(x, y, z, rstride=rstride, cstride=cstride,
                        alpha=texture_alpha, facecolors=img)
    else:
        phi, theta = np.mgrid[0.0:np.pi:32j, 0.0:2.0*np.pi:32j]

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        if texture == 'wire_frame':
            ax.plot_wireframe(x, y, z, cmap=color, alpha=texture_alpha)
        else:
            ax.plot_surface(x, y, z, cmap=color, alpha=texture_alpha)

    if quiver:
        lq = radius * 2
        _x, _y, _z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        _u, _v, _w = [[lq, 0, 0], [0, lq, 0], [0, 0, lq]]
        ax.quiver(_x, _y, _z, _u, _v, _w, color='k', alpha=0.5)

    sphere['x'] = x
    sphere['y'] = y
    sphere['z'] = z

    return sphere


def plot_orbit(orbs, draw_sphere=False, sphere_radius=6378, fig_size=(8, 8),
               show=True, save=None, show_axis=True, projection='persp',
               texture='wire_frame', texture_alpha=0.8, texture_bin=8, title='',
               x_label='X [km]', y_label='Y [km]', z_label='Z [km]',
               elevation=30, azimuth=30):

    if not isinstance(orbs, list):
        orbs = [orbs]

    fig = plt.figure(figsize=fig_size)

    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type(projection)  # 'ortho'

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if not show_axis:
        ax.axis('off')

    if draw_sphere:
        create_sphere(ax, radius=sphere_radius, texture_bin=texture_bin,
                      texture=texture, texture_alpha=texture_alpha)

    colors = plt.cm.jet(np.linspace(0, 1, len(orbs)))

    lim = 0
    legend_handles = list()
    legend_names = list()
    for i, orb in enumerate(orbs):
        x, y, z = orb.ys[:, 0], orb.ys[:, 1], orb.ys[:, 2]

        tmp_lim = np.max([np.abs(item).max() for item in [x, y, z]])
        if tmp_lim > lim:
            lim = tmp_lim

        p, = ax.plot(x, y, z, color=colors[i], lw=1)
        ax.plot(x[0:1], y[0:1], z[0:1], 'ko', ms=5)

        legend_handles.append(p)
        legend_names.append(orb.body.name)

    ax.set_xlim3d([-lim, lim])
    ax.set_ylim3d([-lim, lim])
    ax.set_zlim3d([-lim, lim])

    ax.legend(handles=legend_handles, labels=legend_names)

    ax.view_init(elev=elevation, azim=azimuth)

    if save:
        fig.savefig(save, dpi=300)

    if show:
        plt.show()


def animate_orbit(orbs, draw_sphere=False, sphere_radius=6378, show=True,
                  texture='wire_frame', texture_alpha=0.8, texture_bin=8,
                  show_axis=True, save=False, save_type='gif', dpi=72,
                  filename='animation', rotate=False, rotate_pars=(30, 60),
                  projection='persp', repeat=True, fig_size=(8, 8),
                  x_label='X [km]', y_label='Y [km]', z_label='Z [km]',
                  title='', interval=100, fps=24, bitrate=1800, elevation=30):

    if not isinstance(orbs, list):
        orbs = [orbs]

    fig = plt.figure()  # figsize=fig_size
    fig.set_size_inches(fig_size[0], fig_size[1], True)

    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type(projection)  # 'ortho'

    # ax.w_xaxis.pane.fill = False
    # ax.w_yaxis.pane.fill = False
    # ax.w_zaxis.pane.fill = False
    # ax.grid(False)
    # ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    # ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if not show_axis:
        ax.axis('off')

    lim = 0
    for orb in orbs:
        tmp_lim = np.max([np.abs(item).max() for item in [orb.x, orb.y, orb.z]])
        if tmp_lim > lim:
            lim = tmp_lim

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    ax.tick_params(axis='both', labelsize=8)

    if draw_sphere:
        create_sphere(ax, radius=sphere_radius, texture_bin=texture_bin,
                      texture=texture, texture_alpha=texture_alpha)

    colors = plt.cm.jet(np.linspace(0, 1, len(orbs)))

    particle = []
    trajectory = []
    radial = []
    for i in range(len(orbs)):
        particle.append(ax.plot([], [], [], marker='o', color='k', ms=5)[0])
        trajectory.append(ax.plot([], [], [], color=colors[i], lw=1)[0])
        radial.append(ax.plot([], [], [], color=colors[i], alpha=0.5, lw=1)[0])

    azimuth = np.linspace(rotate_pars[0], rotate_pars[1], orbs[0].n_steps)

    def update(frame):
        for j, orb in enumerate(orbs):
            particle[j].set_data(orb.x[frame], orb.y[frame])
            particle[j].set_3d_properties(orb.z[frame])

            trajectory[j].set_data(orb.x[:frame+1], orb.y[:frame+1])
            trajectory[j].set_3d_properties(orb.z[:frame+1])

            radial[j].set_data([orb.x[frame], 0], [orb.y[frame], 0])
            radial[j].set_3d_properties([orb.z[frame], 0])

        if rotate:
            ax.view_init(elev=elevation, azim=azimuth[frame])

        # return particle, trajectory, radial

    ani = FuncAnimation(fig, update, frames=orbs[0].n_steps, repeat=repeat,
                        interval=interval, blit=False)

    if save:
        if save_type == 'gif':
            ani.save(filename, dpi=dpi, writer='imagemagick')
        else:
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=bitrate)

            ani.save(filename + '.mp4', writer=writer)

    if show:
        plt.show()

    return ani
