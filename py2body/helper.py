#!/usr/bin/env python

__all__ = ['sun', 'earth', 'moon', 'jupiter', 'ecc_anomaly', 'eci2perif']

from numpy import sin, cos, tan, arctan, sqrt, array

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


def kepler(mean_anomaly, e, eps=1e-8, max_iter=100):
    u1 = mean_anomaly

    for _ in range(max_iter):
        u2 = u1 - ((u1 - e * sin(u1) - mean_anomaly) / (1 - e * cos(u1)))

        if abs(u2 - u1) < eps:
            break

        u1 = u2
    else:
        return None

    return u2


def ecc_anomaly(arr, method, eps=1e-8, max_iter=100):
    if method == 'newton':
        Me, e = arr

        return kepler(Me, e, eps=eps, max_iter=max_iter)

    # tae
    ta, e = arr

    return 2 * arctan(sqrt((1 - e) / (1 + e)) * tan(ta / 2))


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
