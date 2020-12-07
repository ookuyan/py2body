#!/usr/bin/env python

__all__ = ['sun', 'earth', 'moon', 'jupiter', 'true_anomaly',
           'eci2perif', 'elem2rv', 'tle2elem']

import datetime

from numpy import sin, cos, tan, arctan, arctan2, sqrt, array
from numpy import deg2rad, rad2deg, pi, transpose, dot

# https://nssdc.gsfc.nasa.gov/planetary/planetfact.html

sun = {
    'name': 'Sun',
    'mass': 1988500e24,
    'radius': 695700.0,
    'mu': 132712e6,
}

earth = {
    'name': 'Earth',
    'mass': 5.972e24,
    'radius': 6378.0,
    'mu': 0.39860e6,
    'J2': -1.082635854e-3
}

moon = {
    'name': 'Moon',
    'mass': 0.07346e24,
    'radius': 1738.1,
    'mu': 0.00490e6,
}

jupiter = {
    'name': 'Jupiter',
    'mass': 1898.19e24,
    'radius': 69911,
    'mu': 126.687e6,
}


def ecc_anomaly(e, M, eps=1e-8, max_iter=100):
    u1 = M

    for _ in range(max_iter):
        u2 = u1 - ((u1 - e * sin(u1) - M) / (1 - e * cos(u1)))

        if abs(u2 - u1) < eps:
            break

        u1 = u2
    else:
        return None

    return u2


def true_anomaly(e, E):
    return 2 * arctan2(sqrt(1 + e) * sin(E / 2), sqrt(1 + e) * cos(E / 2))


def eci2perif(lan, aop, i):
    u = [
        -sin(lan) * cos(i) * sin(aop) + cos(lan) * cos(aop),
        cos(lan) * cos(i) * sin(aop) + sin(lan) * cos(aop),
        sin(i) * sin(aop)
    ]
    v = [
        -sin(lan) * cos(i) * cos(aop) - cos(lan) * sin(aop),
        cos(lan) * cos(i) * cos(aop) - sin(lan) * sin(aop),
        sin(i) * cos(aop)
    ]
    w = [
        sin(lan) * sin(i),
        -cos(lan) * sin(i),
        cos(i)
    ]

    return array([u, v, w])


def elem2rv(elements, mu=earth['mu']):
    a, e = elements['a'], elements['e']
    i = deg2rad(elements['i'])
    aop = deg2rad(elements['argument_of_periapsis'])
    lan = deg2rad(elements['longitude_of_ascending_node'])

    E = ecc_anomaly(e=e, M=elements['mean_anomaly'])
    ta = true_anomaly(e=e, E=E)

    r_norm = a * (1 - e * cos(E))
    r_perif = r_norm * array([cos(ta), sin(ta), 0])
    v_perif = sqrt(mu * a) / r_norm
    v_perif *= array([-sin(E), cos(E) * sqrt(1 - e ** 2), 0])

    perif2eci = transpose(eci2perif(lan, aop, i))

    position = dot(perif2eci, r_perif).tolist()
    velocity = dot(perif2eci, v_perif).tolist()

    period = 2 * pi * sqrt(a ** 3 / mu)

    return position, velocity, period


def calc_epoch(epoch, year_prefix='20'):
    year = int(year_prefix + epoch[:2])

    epoch = epoch[2:].split('.')

    day_of_year = int(epoch[0]) - 1
    hour = float('0.' + epoch[1]) * 24.0
    date = datetime.date(year, 1, 1) + datetime.timedelta(day_of_year)

    month = int(date.month)
    day = int(date.day)

    return year, month, day, hour


def tle2elem(tle):
    line0, line1, line2 = tle

    line0 = line0.strip()
    line1 = line1.strip().split()
    line2 = line2.strip().split()

    # epochs
    epoch = line1[3]
    year, month, day, hour = calc_epoch(epoch)

    # inclination
    i = float(line2[2])

    # right ascention of ascending node / longitude of ascending node
    lan = float(line2[2])

    # eccentricity
    e = float('0.' + line2[4])

    # argument of periapsis
    aop = float(line2[5])

    # mean anomaly
    M = float(line2[6])

    # mean motion [revs / day]
    mean_motion = float(line2[7])

    # period [seconds]
    period = 1 / mean_motion * 86400

    # semi major axis
    a = (period**2 * earth['mu'] / 4.0 / pi**2) ** (1 / 3)

    # eccentric anomaly
    E = ecc_anomaly(e, deg2rad(M))

    # true anomaly
    ta = true_anomaly(E, e)

    elements = dict()
    elements['name'] = line0
    elements['a'] = a
    elements['e'] = e
    elements['i'] = i
    elements['true_anomaly'] = rad2deg(ta)
    elements['mean_anomaly'] = M
    elements['argument_of_periapsis'] = aop
    elements['longitude_of_ascending_node'] = lan
    elements['period'] = period
    elements['epoch'] = {'year': year, 'month': month, 'day': day, 'hour': hour}

    return elements
