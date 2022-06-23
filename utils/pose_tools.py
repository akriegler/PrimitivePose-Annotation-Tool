"""
    FUNCTION: pose_tools.py
----------------------------------------------------------------------------------------------------------
    DESCRIPTION:
        All functions related to camera and object poses
----------------------------------------------------------------------------------------------------------
    RESPONSIBILITY: A. Kriegler (Andreas.Kriegler@ait.ac.at)
    VERSIONs:       python 3.7
    CREATION DATE:  2021/08/18
----------------------------------------------------------------------------------------------------------
    AIT - Austrian Institute of Technology
----------------------------------------------------------------------------------------------------------
"""
import copy
import math

import bpy
import numpy as np

from numpy.linalg import inv
from mathutils import Vector, Matrix, Quaternion

from scipy.spatial.transform import Rotation as Rotfunc

import cam_tools as ctools


def get_2d_3d_bounding_box(cam, obj, width, height):
    """
    Determining 3D model coordinates (box_crds) and image bounding box coordinates [ulx, uly, brx, bry]
    also, at this point we computed the image BB center: center_cxy
    """
    ulx = 9999
    uly = 9999
    brx = -9999
    bry = -9999

    for b in obj.bound_box:
        box_vec = Vector(b)
        world_crds = obj.matrix_world @ box_vec  # world space coordinates -> not needed
        cur_xy = worldpoint_to_pixel(cam, np.asarray(world_crds))  # converting the current 3D point to an image pixel
        cx = cur_xy[0]
        cy = cur_xy[1]
        if cx < ulx:  # determining minimum and maximum
            ulx = cx
        if cx > brx:
            brx = cx
        if cy < uly:
            uly = cy
        if cy > bry:
            bry = cy
    ulx = max(0, ulx)
    uly = max(0, uly)
    brx = min(width, brx)
    bry = min(height, bry)

    # returning the 2d bounding box corners
    return ulx, uly, brx, bry


def get_6dof_pose_params(cam, obj_type, obj, fx, fy, px, py):
    """
        Determining the 6DoF pose parameters for a given (currobj) obj. seen by the camera cam
        This function computes multiple directly observable variables:
            2D rectangle (bounding box)
            translation vector (object-to-camera), directly estimating from the object 2D center given a distance
                (depth) and cam. intrinsics

        INPUTS:
            currobj  - current selection of viewed object
            cam      - camera observer
            fx, fy   - focal length (corrected by fill-factor,etc. factors)  /determined from camera matrix K/
            px, py   - principal points along x and y                        /determined from camera matrix K/
            im_width - image width [pixel]
        OUTPUTS:
            centx, centy                    - image-back-projected object center        [pixel] /can be regressed/
            ulx, uly, brx, bry              - 2D bounding box definition                [pixel] /can be regressed/
            xdim, ydim, zdim                - 3D dimensions: width, length, height                 [meter]
            transl_vec                      - object translation vector (relative to the camera)   [meter]
    """
    # --- determining the 3D center of the current object, then back-projecting it into the image
    obj_cent = get_object_center(obj)  # determining the 3D location of the obj. center
    center_cxy = worldpoint_to_pixel(cam, obj_cent)  # determining the 2D location of the obj. center
    centx = center_cxy[0]  # back-projected center x- and y-coordinates
    centy = center_cxy[1]

    # determining the 2D (rectangular) bounding box coordinates
    ulx, uly, brx, bry = get_2d_3d_bounding_box(cam, obj, int(px * 2), int(py * 2))

    # --- determining object dimensions --> these values correspond to the dim. params in the Modeling view

    xdim = obj.dimensions[0]
    ydim = obj.dimensions[1]
    zdim = obj.dimensions[2]

    # --- determining the relative orientation of the object (w.r.t to the camera) and the depth (z-distance) from
    # the camera
    res_rot, delta_z = get_obj_orientation(cam, obj)

    # --- computing the translation vector from the object center to the camera center (projective similarity)
    transl_vec = calculate_translation_from_cxy(centx, centy, delta_z, fx, fy, px, py)

    # --- Output variables: ------------------------------------------------------------------------------------
    # centx, centy                    - image-back-projected object center                               [pixel]
    # ulx, uly, brx, bry              - 2D bounding box definition                                       [pixel]
    # xdim, ydim, zdim                - 3D dimensions: width, length, height                             [meter]
    # transl_vec                      - object translation vector (relative to the camera)               [meter]
    # delta-z                         - z-depth (distance object from camera)                            [meter]
    #    return [centx, centy, ulx, uly, brx, bry, xdim, ydim, zdim, transl_vec, delta_z]
    return [centx, centy, ulx, uly, brx, bry, xdim, ydim, zdim, transl_vec, delta_z]


def calculate_translation_from_cxy(centx, centy, depth, fx, fy, px, py):
    """
        computes the camera translation vector based on an image centroid and its depth
    """
    # Tx = (cx - px) * Tz / fx
    # Ty = (cy - py) * Tz / fy
    # cx, cy --> object center in image
    # px, py --> image principal points
    # fx, fy --> focal length

    # print("====================")
    # print(centx, centy, depth, px, py)

    tx = (centx - px) * depth / fx
    ty = (centy - py) * depth / fy
    tz = depth

    return np.asarray([tx, ty, tz])


def compute_box_3d(dim, location, rotation_x, rotation_y, rotation_z):
    """
        Computes the 3D corners coordinates (in an object-centered reference) by
            - creating a cuboid with dimensions of l, w, and h [unit: m] around a center of (0,0,0)
            - applying three rotations given rotx, roty, and rotz angles [unit: radian]
        Inputs:
            dim: cuboid dimensions [length, width, height]
            location: translation vector [dx, dy, dz] w.r.t. the camera
            rotation_x, rotation_y, rotation_z: rotational angles [radian] relative to camera [axis-separated angles]
        Output:
            returns 8 x 3 (x,y,z) cuboid corner coordinates within the camera 3D space
    """
    # rotation matrix for rotation around x
    cx, sx = np.cos(rotation_x), np.sin(rotation_x)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)

    # rotation matrix for rotation around y
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    # rotation matrix for rotation around z
    cz, sz = np.cos(rotation_z), np.sin(rotation_z)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    l, w, h = dim[0], dim[1], dim[2]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2]
    z_corners = [h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)

    # performing rotation for each angle
    corners_3d = np.dot(rx, corners)
    corners_3d = np.dot(ry, corners_3d)
    corners_3d = np.dot(rz, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)

    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, k):
    """
        back-projecting a set of 3D (pre-rotated + translated) points into the image using the K camera matrix
        Inputs:
            pts_3d: n x 3 3D coordinates
            K: 3 x 4 camera matrix
        Output:
            pts_2d: n x 2 image points, defined by [x,y] coordinates
    """
    #  adding zero-valued coloumn to the camera matrix
    k = np.concatenate([k, np.zeros((k.shape[1], 1), dtype=np.float32)], axis=1)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(k, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d


def get_obj_orientation(cam, obj):
    """
        A core function to determine the rel. orientation (euler angles) with respect to the camera.
        The function performs following steps:
           STEP1: saves object location/rotation and camera rotation parameters
           STEP2: parents the object to the camera. with keep transform (= when camera is rotated
                  the object also rotates with it
           STEP3: the camera is rotated into a (0,0,0) orientation
           STEP4: the object is unparented from the camera, and its zdepth (z-distance) and its Euler orientation values
           are read out
           STEP5: object position/orientation and camera orientation are restored
           INPUT:
               currobj - the current (MESH) object whose relative rotational parameters and depth distance from camera
               are to be determined
               cam     - camera object (observer)
           OUTPUT:
               res_rot - Euler angles representing the relative obhect orientation w.r.t. the camera
               delta_z - z-distance (depth) of the object from camera center (later needed to compute the translation
               vector given the object center in image)
    """

    # --- STEP1: storing object spatial attributes ----------------------------------------------------
    obj_rot_cp = copy.deepcopy(obj.rotation_euler)
    obj_loc_cp = copy.deepcopy(obj.location)

    # camera orientation
    cam_rot_cp = copy.deepcopy(cam.rotation_euler)

    # --- STEP2: parenting the object to the camera ---------------------------------------------------
    bpy.ops.object.select_all(action='DESELECT')  # deselecting all objects
    obj.select_set(True)  # select the object as the 'child'
    cam.select_set(True)  # select the object as the 'parent'

    bpy.context.view_layer.objects.active = cam
    bpy.ops.object.parent_set(type='OBJECT', xmirror=False, keep_transform=True)

    # --- STEP3: rotating the camera to a reference position ------------------------------------------
    cam.rotation_euler[0] = 0.0
    cam.rotation_euler[1] = 0.0
    cam.rotation_euler[2] = 0.0

    # --- STEP4: unparenting with keep transform ------------------------------------------------------
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # determing the z-depth between object and camera centers
    delta_z = obj.location[2] - cam.location[2]

    # retrieving the current object orientation (quantity which is returned)
    res_rot = copy.deepcopy(obj.rotation_euler)

    # --- STEP5: restoring the object location and rotation -------------------------------------------
    obj.location = obj_loc_cp
    obj.rotation_euler = obj_rot_cp

    # restoring the camera
    cam.rotation_euler = cam_rot_cp

    bpy.ops.object.select_all(action='DESELECT')  # deselecting all objects

    return res_rot, delta_z


def get_rel_orient_to_camera(cam, obj_type, currobj, fx, fy, px, py, im_width, force_symmetry_angle_zero,
                             force_planar_angle_zero):
    """
    Function computing first the relative object (currobj) orientation w.r.t. the camera.
    In a second step, the apparent orientations (*_rel) are computed, based on the object location in the image
    Inputs:
        cam - camera object
        currobj  - reference to the object
        fx, fy   - focal length (*sensor corrective factors) taken from the camera instrinsic matrix
        px, py   - principal points taken from the camera instrinsic matrix
        im_width - the width of the camera image [unit: pixels]
    Outputs:
        x_ang_rel, y_ang_rel, z_ang_rel - apparent relative orientations with respect to the viewer (camera)
    """
    # --- determining the 3D center of the current object, then back-projecting it into the image
    obj_cent = get_object_center(currobj)  # determining the 3D location of the obj. center
    center_cxy = worldpoint_to_pixel(cam, obj_cent)  # determining the 2D location of the obj. center
    centx = center_cxy[0]  # back-projected center x- and y-coordinates
    centy = center_cxy[1]

    # camera world orientation matrix
    location_cam, rotation = cam.matrix_world.decompose()[0:2]
    r_world2bcam = rotation.to_matrix().transposed()

    # object world orientation matrix
    location_obj, rotation = currobj.matrix_world.decompose()[0:2]

    obj_world2bcam = rotation.to_matrix().transposed()

    # computing the relative orientation: object with respect to camera
    # this relative orientation correctly decouples per-axis rotational components
    # to re-project the rotated shape in the camera, a rot. matrix needs to be inverted and used for reconstruction
    relormat_o2c = obj_world2bcam @ r_world2bcam.inverted()  # these orientations well represent the 3 orientations
    rel_angles_euler = relormat_o2c.to_euler()

    x_ang_obj = rel_angles_euler[0]
    y_ang_obj = -rel_angles_euler[2]  # NOTE: changing axis order -> XZY
    z_ang_obj = -rel_angles_euler[1]

    # upon reconstruction:
    # first we estimate the visible orientations
    # second: we convert them to absolute orientations
    # finally, we generate a corresponding rotation matrix, invert and use for re-projection

    x_ang_rel = rot2alpha(x_ang_obj, centy, py, fy)
    # inputs: abs. or., cx = obj center, px = princ. p, fx = focal length
    y_ang_rel = rot2alpha(y_ang_obj, im_width - centx, px, fx)
    z_ang_rel = clamp_rot(z_ang_obj)

    if force_symmetry_angle_zero:
        if 'sph' in obj_type or 'Con' in obj_type or 'Cyl' in obj_type:
            y_ang_rel = 0.0
    if force_planar_angle_zero:
        z_ang_rel = 0.0

    return x_ang_rel, y_ang_rel, z_ang_rel


def get_shortest_dist(obj, cam, shortest_dist_arr, x_range, y_range, bb, total_obj_idx, u_l):
    """
        https://blender.stackexchange.com/questions/115285/how-to-do-a-ray-cast-from-camera-originposition-to-object-in-scene-in-such-a-w
    """
    # save current view mode
    # set view mode to 3D to have all needed variables available
    old_view = bpy.context.area.type
    bpy.context.area.type = "VIEW_3D"

    # iterate over all X/Y coordinates
    # calculate origin
    world_matrix = obj.matrix_world
    world_matrix_inv = world_matrix.inverted()
    origin = world_matrix_inv @ cam.matrix_world.translation
    # reset indices
    x_pix_start = max(0, round(bb[0]))
    x_pix_end = min(len(x_range), round(bb[2]))
    y_pix_start = max(0, round(bb[1]))
    y_pix_end = min(len(y_range), round(bb[3]))
    x_ind = x_pix_start
    y_ind = y_pix_start
    y_min = 9999
    y_max = 0
    x_min = 9999
    x_max = 0

    for y in y_range[y_pix_start: y_pix_end]:
        for x in x_range[x_pix_start: x_pix_end]:
            # get current pixel vector from camera center to pixel
            pix_vec = Vector((x, y, u_l[2]))
            # rotate that vector according to camera rotation
            pix_vec.rotate(cam.matrix_world.to_quaternion())
            # calculate direction vector
            destination = world_matrix_inv @ (pix_vec + cam.matrix_world.translation)
            direction = (destination - origin).normalized()
            # perform the actual ray casting
            hit, location, norm, face = obj.ray_cast(origin, direction)
            if hit:
                distance_from_camera = (cam.matrix_world.translation - (world_matrix @ location)).length
                shortest_dist_arr[total_obj_idx, y_ind, x_ind] = distance_from_camera
                if x_ind < x_min:
                    x_min = x_ind
                if x_ind > x_max:
                    x_max = x_ind
                if y_ind < y_min:
                    y_min = y_ind
                if y_ind > y_max:
                    y_max = y_ind

            # update indices
            x_ind += 1
        y_ind += 1
        x_ind = max(0, round(bb[0]))
    bpy.context.area.type = old_view

    ul_x = max(x_min - 1, 0)
    ul_y = max(y_min - 1, 0)
    lr_x = min(x_max + 1, len(x_range))
    lr_y = min(y_max + 1, len(y_range))

    return shortest_dist_arr, ul_x, ul_y, lr_x, lr_y


def worldpoint_to_pixel(cam, w_coord):
    """
        a world point [x,y,z] to a given image pixel
        Converts all vertices of the objects into subpixel coordinates.
    """
    p_matrix, _, _ = ctools.get_3x4_p_matrix_from_blender(cam)
    pc = np.array(p_matrix) @ np.array([w_coord[0], w_coord[1], w_coord[2], 1])
    pc = dehom(pc)

    return np.asarray(pc)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTION decode_orientations -
# ----------------------------------------------------------------------------------------------------------------------------------------------------
def decode_orientations(x_ang_rel, y_ang_rel, z_ang_rel, center_cxy, fx, fy, px, py, res_x):
    """
        Generating the object-centered orientations via a series of transforms
        (using scipy.spatial.transform import Rotation)
        Following transformation steps are executed:
            Step 1: from visible orientations in object-centered ref.  -> absolute orientations in object-centered ref.
            Step 2: from absolute orientations in object-centered ref. -> absolute orientation in camera reference
        Inputs:
            x_ang_rel, y_ang_rel, z_ang_rel: elevation, azimuthal and roll (default: 0.0) angles [unit: radian]
            center_cxy: object center coordinates in the image [x,y]. Computed via back-projecting the 3D object center
            into the image [unit: pixels]
            fx, fy: focal length along x and y, from the camera intrinsics
            px, py: principal point along x and y, from the camera intrinsics
        Outputs:
            x_angt, y_angt, z_angt: object orientation in camera reference [unit: radian]
    """
    # orientation conversion steps: from visible orientations in object-centered ref.
    # -> absolute orientations in object-centered ref.
    # -> absolute orientation in camera reference
    x_ang = alpha2rot(x_ang_rel, center_cxy[1], py, fy)
    y_ang = alpha2rot(y_ang_rel, res_x - center_cxy[0], px, fx)
    z_ang = z_ang_rel

    # computing the absolute orientations in degree
    x_ang_degr = x_ang * 180 / math.pi
    y_ang_degr = -z_ang * 180 / math.pi
    z_ang_degr = -y_ang * 180 / math.pi

    rotmat = Rotfunc.from_euler('xyz', [x_ang_degr, y_ang_degr, z_ang_degr], degrees=True)
    testmat = rotmat.as_matrix()
    invrotmat = inv(testmat)  # inversion switches the relationship: object-to-camera --> camera-to-object

    rotfinal = Rotfunc.from_matrix(invrotmat)
    eul_angles = rotfinal.as_euler('xyz', degrees=True)

    # computing the output orienation values (object orientation in camera ref.)
    x_angt = eul_angles[0] * math.pi / 180.0
    y_angt = -eul_angles[1] * math.pi / 180.0
    z_angt = -eul_angles[2] * math.pi / 180.0

    return x_angt, y_angt, z_angt, x_ang, y_ang, z_ang


def clamp_rot(rot):
    """
        Ensures that all rotation angles are within [-pi, pi] to solve ambiguities (singularities)
    """
    if rot > (np.pi / 2.0):
        rot -= np.pi
    if rot < -(np.pi / 2.0):
        rot += np.pi

    return rot


def alpha2rot(alpha, obj_center, principal_point, focal_length):
    """
        Get rotation by alpha + theta (shifted to 2pi range)
    """
    rot = alpha + np.arctan2(obj_center - principal_point, focal_length)
    rot = clamp_rot(rot)

    return rot


def rot2alpha(rot, obj_center, principal_point, focal_length):
    """
        Get alpha by rotation - theta (shifted to 2pi range)
    """
    alpha = rot - np.arctan2(obj_center - principal_point, focal_length)
    alpha = clamp_rot(alpha)

    return alpha


def dehom(x):
    """
        Return de-homogeneous coordinates by perspective division.
        Denormalizes and removes a row of values.

        # >>> a = np.array([[  0,   1,   2],
                            [100, 101, 102],
                            [  1,   1,   1]])
        # >>> dehom(a)
                    array([[  0.,   1.,   2.],
                           [100., 101., 102.]])
    """

    return x[:-1, ...] / x[-1:, ...]


def get_object_center(obj):
    """
        determining the center of an object by computing the mean coordinates of its 3D bounding box
    """
    loc_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    loc_bbox_center = obj.matrix_world @ loc_bbox_center

    return loc_bbox_center
