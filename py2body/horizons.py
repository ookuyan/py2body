#!/usr/bin/env python

__all__ = ['get_body_from_horizon']


from astroquery.jplhorizons import Horizons


G = 6.67408e-20  # units of km^3/kg/s^2


def get_body_from_horizon(name, epochs=None, body_type='majorbody',
                          mass=None, plane='ecliptic'):

    if not isinstance(name, str):
        raise TypeError("'name' should be a 'str' object.")

    if not isinstance(epochs, (type(None), dict)):
        raise TypeError("'date' should be 'None' or 'dict' object.")

    if not isinstance(body_type, str):
        raise TypeError("'body_type' should be a 'str' object.")

    if body_type not in ['majorbody', 'smallbody']:
        raise ValueError("'body_type' name should be "
                         "'majorbody' or 'smallbody'.")

    if not isinstance(mass, (type(None), float)):
        raise TypeError("'mass' should be 'None' or 'float' object.")

    if not isinstance(plane, str):
        raise TypeError("'plane' should be a 'str' object.")

    if plane not in ['ecliptic', 'earth']:
        raise ValueError("'plane' name should be 'ecliptic' or 'earth'.")

    obj = Horizons(id=name, epochs=epochs, id_type=body_type)

    try:
        t = obj.vectors(refplane=plane)
    except ValueError:
        return None

    if mass is None:
        if body_type == 'majorbody':
            try:
                mass = float(obj.raw_response[
                             obj.raw_response.find('GM'):]
                             .split('\n')[0].split()[3]) / G
            except IndexError:
                mass = 0
            except ValueError:
                mass = 0
        else:
            try:
                mass = float(obj.raw_response[
                             obj.raw_response.find('GM='):]
                             .split('\n')[0].split()[1]) / G
            except IndexError:
                mass = 0

    if len(t) == 1:
        date = str(t['datetime_str'][0])
        jd = float(t['datetime_jd'][0])
        x, y, z = float(t['x'][0]), float(t['y'][0]), float(t['z'][0])
        vx, vy, vz = float(t['vx'][0]), float(t['vy'][0]), float(t['vz'][0])
    else:
        jd = t['datetime_jd'].data.data
        date = t['datetime_str'].data.data
        x, y, z = t['x'].data.data, t['y'].data.data, t['z'].data.data
        vx, vy, vz = t['vx'].data.data, t['vy'].data.data, t['vz'].data.data

    body = {
        'name': str(t['targetname'][0]), 'mass': mass, 'date': date, 'jd': jd,
        'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz,
        'pos_unit': 'au', 'vel_unit': 'au/d'
    }

    return body
