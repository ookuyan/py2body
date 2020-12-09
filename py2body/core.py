#!/usr/bin/env python

__all__ = ['Body', 'Orbit', 'sun', 'earth', 'moon', 'jupiter']

import numpy as np
from scipy.integrate import ode

from .helper import *


class Body(object):

    def __init__(self, **args):
        self.name = 'Satellite'
        self.mu = earth['mu']
        self.mass = 100.0

        self.position = None
        self.velocity = None

        self.elements = None
        self.period = None

        self.epoch = T.now()

        self.tle = None

        self.Cd = 2.2
        self.area = 1e-3**2 / 4

        if 'name' in args:
            self.name = args['name']

        if 'mu' in args:
            self.mu = args['mu']

        if 'mass' in args:
            self.mass = args['mass']

        if 'epoch' in args:
            self.epoch = args['epoch']

        if 'tle' in args:
            self.tle = args['tle']
            self.elements = tle2elem(self.tle)

            self.position, self.velocity, self.period = \
                elem2rv(elements=self.elements, mu=self.mu)

            self.name = self.elements['name']
            self.epoch = self.elements['epoch']

        if 'elements' in args:
            self.position, self.velocity, self.period = \
                elem2rv(elements=args['elements'], mu=self.mu)

            self.elements = args['elements']

        if 'position' in args:
            self.position = args['position']
        else:
            if self.position is None:
                r_mag = earth['radius'] + 408
                self.position = [r_mag, 0, 0]

        if 'velocity' in args:
            self.velocity = args['velocity']
        else:
            if self.velocity is None:
                v_mag = np.sqrt(self.mu / self.position[0])  # circular
                self.velocity = [0, v_mag, 0]

        if 'Cd' in args:
            self.Cd = args['Cd']

        if 'area' in args:
            self.A = args['area']


class Orbit(object):

    def __init__(self, body, center=earth, dt=60, n_steps=100,
                 integrator='dop853'):  # lsoda

        self.body = body
        self.center = center
        self.mu = self.center['mu']

        self.dt = dt
        self.step = 0
        self.n_steps = n_steps

        self.integrator = integrator

        self.ts = None
        self.ts_ = None
        self.ys = None

        self.solver = ode(self.f)
        self.solver.set_integrator(integrator)

        self.x = None
        self.y = None
        self.z = None
        self.vx = None
        self.vy = None
        self.vz = None

        self.elements = None
        self.altitude = None

        self.perturbations = {
            'j2': False, 'drag': False, 'srp': False,
            'sun': False, 'moon': False, 'jupiter': False
        }

    def set_perturbations(self, **args):
        for key, val in args.items():
            if key in self.perturbations:
                self.perturbations[key] = val

    def f(self, t, y):
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        v = np.array([vx, vy, vz])

        norm_r = np.linalg.norm(r)
        a = -r * self.mu / norm_r ** 3

        if self.perturbations['j2']:
            z2 = r[2] ** 2
            r2 = norm_r ** 2
            tx = r[0] / norm_r * (5 * z2 / r2 - 1)
            ty = r[1] / norm_r * (5 * z2 / r2 - 1)
            tz = r[2] / norm_r * (5 * z2 / r2 - 3)

            a_j2 = 1.5 * self.center['j2'] * self.mu * \
                self.center['radius']**2 / norm_r**4 * \
                np.array([tx, ty, tz])

            a += a_j2

        if self.perturbations['drag']:
            z = norm_r - self.center['radius']
            rho = calc_atmospheric_density(z, self.center)

            v_rel = v - np.cross(self.center['atm']['rot_vector'], r)
            drag = -v_rel * 0.5 * rho * np.linalg.norm(v_rel) *\
                self.body.Cd * self.body.area / self.body.mass

            a += drag

        # t_ = T.from_datetimes(
        #     self.body.epoch.utc_datetime() + timedelta(seconds=t))
        # cb = self.center['eph']

        # if self.perturbations['sun']:
        #     tb = sun['eph']
        #     rs = cb.at(t_).observe(tb).position.km
        #
        #     norm_rs = np.linalg.norm(rs)
        #     a += -rs * sun['mu'] / norm_rs ** 3
        #
        # if self.perturbations['moon']:
        #     tb = moon['eph']
        #     rm = cb.at(t_).observe(tb).position.km
        #
        #     norm_rm = np.linalg.norm(rm)
        #     a += -rm * moon['mu'] / norm_rm ** 3
        #
        # if self.perturbations['jupiter']:
        #     tb = jupiter['eph']
        #     rj = cb.at(t_).observe(tb).position.km
        #
        #     norm_rj = np.linalg.norm(rj)
        #     a += -rj * jupiter['mu'] / norm_rj ** 3

        ax, ay, az = a

        # self.ts_.append(t_)

        return [vx, vy, vz, ax, ay, az]

    def propagate(self):
        self.step = 0
        self.elements = list()
        self.altitude = list()
        self.ts_ = list()
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

            self.elements.append(
                rv2elem(self.ys[self.step][:3], self.ys[self.step][3:],
                        mu=self.center['mu']))

            self.altitude.append(
                np.linalg.norm(self.ys[self.step][:3]) - self.center['radius'])

            self.step += 1

        self.x, self.y, self.z = self.ys[:, 0], self.ys[:, 1], self.ys[:, 2]
        self.vx, self.vy, self.vz = self.ys[:, 3], self.ys[:, 4], self.ys[:, 5]

        self.altitude = np.array(self.altitude)
