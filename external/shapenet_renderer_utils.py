# https://github.com/vsitzmann/shapenet_renderer
import numpy as np
import math


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


# All the following functions follow the opencv convention for camera coordinates.
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    up = np.array([0., 0., 1.])

    right = np.cross(forward, up)
    right = normalize(right)

    up = np.cross(right, forward)
    up = normalize(up)

    mat = np.stack((right, up, -forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def sample_spherical(n, radius=1.):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    return xyz


def get_archimedean_spiral(num_steps, sphere_radius=1.):
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []

    i = a / 2
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        y = r * math.sin(-theta + math.pi) * math.sin(-i)
        z = - r * math.cos(theta)

        translations.append((x, y, z))
        i += a / (2 * num_steps)

    return np.array(translations)
