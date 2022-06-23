"""
    FUNCTION: cam_tools.py
----------------------------------------------------------------------------------------------------------
    DESCRIPTION:
        Function performing camera manipulation (look at target) and conversion (view -> image) functions
----------------------------------------------------------------------------------------------------------
    RESPONSIBILITY: A. Kriegler (Andreas.Kriegler@ait.ac.at)
    VERSIONs:       python 3.7
    CREATION DATE:  2021/08/18
----------------------------------------------------------------------------------------------------------
    AIT - Austrian Institute of Technology
----------------------------------------------------------------------------------------------------------
"""

import bpy

import numpy as np

from math import *

from mathutils import Vector, Matrix


def get_blender_rt_matrix(cam):
    """
        Returns camera rotation and translation matrices from Blender
        There are 3 coordinate systems involved:
            1. The World coordinates: "world"
                - right-handed
            2. The Blender camera coordinates: "bcam"
                - x is horizontal
                - y is up
                - right-handed: negative z look-at direction
            3. The desired computer vision camera coordinates: "cv"
                - x is horizontal
                - y is down (to align to the actual pixel coordinates used in digital images)
                - right-handed: positive z look-at direction
    """
    # bcam stands for blender camera
    r_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    r_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    t_world2bcam = -1 * r_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    r_world2cv = r_bcam2cv @ r_world2bcam
    t_world2cv = r_bcam2cv @ t_world2bcam

    # put into 3x4 matrix
    rt = Matrix(
        (
            r_world2cv[0][:] + (t_world2cv[0],),
            r_world2cv[1][:] + (t_world2cv[1],),
            r_world2cv[2][:] + (t_world2cv[2],),
        )
    )

    return rt


def get_blender_k_matrix(cam_data):
    """
        Build intrinsic camera parameters from Blender camera data

        See notes on this in
        blender.stack_matrixexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
        as well as
        https://blender.stack_matrixexchange.com/a/120063/3581
        https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera

        for <output_convention_pixel_origin_at_pixel_center> see objects_to_pixels
    """
    output_convention_pixel_origin_at_pixel_center = True

    if cam_data.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = cam_data.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y

    sensor_size_in_mm = cam_data.sensor_height if cam_data.sensor_fit == 'VERTICAL' else cam_data.sensor_width
    sensor_fit = get_sensor_fit(cam_data.sensor_fit, scene.render.pixel_aspect_x * resolution_x_in_px,
                                scene.render.pixel_aspect_y * resolution_y_in_px)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix k_matrix
    u_0 = (resolution_x_in_px / 2) - (cam_data.shift_x * view_fac_in_px)
    v_0 = (resolution_y_in_px / 2) + (cam_data.shift_y * view_fac_in_px / pixel_aspect_ratio)

    if output_convention_pixel_origin_at_pixel_center:
        u_0 -= 0.5
        v_0 -= 0.5

    # only use rectangular pixels
    skew = 0

    k_matrix = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))

    return k_matrix


def get_3x4_p_matrix_from_blender(cam_object):
    """
    Returns the projection matrix P(3x4), Calibration Matrix K(3x3) and
        the [Rotation (3x3)|translation (3x1)] Rt(3x4) matrix
    P, K, Rt = get_3x4_P_matrix_from_blender(cam_object)

    """
    k_matrix = get_blender_k_matrix(cam_object.data)
    rt = get_blender_rt_matrix(cam_object)

    return k_matrix @ rt, k_matrix, rt


def get_sensor_fit(sensor_fit, size_x, size_y):
    """
        BKE_camera_sensor_fit
        Reproduces the python logic for sensor_fit = 'AUTO'
    """
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit
