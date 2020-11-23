#!/usr/bin/env python

__all__ = ['Body', 'Orbit', 'sun', 'earth', 'moon', 'jupiter']

import numpy as np
from scipy.integrate import ode

from .helper import *


class Body(object):

    def __init__(self, name='Sat', elements=None, position=None,
                 velocity=None, mu=None, mass=None):

        self.name = name
        self.mu = mu

        if mu is None:
            self.mu = earth['mu']

        self.period = None

        if elements is None:
            self.position = position
            self.velocity = velocity

            if position is None:
                r_mag = earth['radius'] + 408
                self.position = [r_mag, 0, 0]

            if velocity is None:
                v_mag = np.sqrt(earth['mu'] / self.position[0])  # circular
                self.velocity = [0, v_mag, 0]
        else:
            self.position, self.velocity, self.period = \
                elem2rv(elements=elements, mu=self.mu)

        if mass is None:
            self.mass = 0.0


class Orbit(object):

    def __init__(self, body, mu=None, dt=60, n_steps=100, integrator='lsoda'):
        self.body = body
        self.mu = mu

        if mu is None:
            self.mu = body.mu

        self.dt = dt
        self.step = 0
        self.n_steps = n_steps

        self.integrator = integrator

        self.ts = None
        self.ys = None

        self.solver = ode(self.f)
        self.solver.set_integrator(integrator)

        self.x = None
        self.y = None
        self.z = None
        self.vx = None
        self.vy = None
        self.vz = None

    def f(self, t, y):
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])

        n_r = np.linalg.norm(r)
        ax, ay, az = -r * self.mu / n_r**3

        return [vx, vy, vz, ax, ay, az]

    def propagate(self):
        self.ts = np.zeros((self.n_steps, 1))
        self.ys = np.zeros((self.n_steps, 6))

        y0 = self.body.position + self.body.velocity
        self.solver.set_initial_value(y0, 0)
        self.ys[0] = np.array(y0)
        self.step += 1

        while self.solver.successful() and self.step < self.n_steps:
            self.solver.integrate(self.solver.t + self.dt)

            self.ts[self.step] = self.solver.t
            self.ys[self.step] = self.solver.y

            self.step += 1

        self.x, self.y, self.z = self.ys[:, 0], self.ys[:, 1], self.ys[:, 2]
