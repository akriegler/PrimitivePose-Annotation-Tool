"""
    FUNCTION: results_tools.py
----------------------------------------------------------------------------------------------------------
    DESCRIPTION:
        Utility function for various results saving functionalities
----------------------------------------------------------------------------------------------------------
    RESPONSIBILITY: A. Kriegler (Andreas.Kriegler@ait.ac.at)
    VERSIONs:       python 3.7
    CREATION DATE:  2021/05/10
----------------------------------------------------------------------------------------------------------
    AIT - Austrian Institute of Technology
----------------------------------------------------------------------------------------------------------
"""

import os
import sys
import math
import glob

import cv2
import bpy

import numpy as np

from tqdm import tqdm
from pathlib import Path

from easydict import EasyDict as edict


def relpath(path_to, path_from):
    path_to = Path(path_to).resolve()
    path_from = Path(path_from).resolve()

    head = None
    tail = None

    try:
        for p in (*reversed(path_from.parents), path_from):
            head, tail = p, path_to.relative_to(p)
    except ValueError:  # Stop when the paths diverge.
        pass
    return Path('../' * (len(path_from.parents) - len(head.parents))).joinpath(tail)


def generate_dir_structure(params):
    """
    Generating a directory structure containing annotation files and debug visualizations
    """
    if not os.path.exists(params['out_path']):
        print(" --- Created output directory: {} ---".format(params['out_path']))
        os.makedirs(params['out_path'])

    if not os.path.exists(params['pose_gt_path']):
        print(" --- Created ground truth output directory: {} ---".format(params['pose_gt_path']))
        os.makedirs(params['pose_gt_path'])

    if not os.path.exists(params['pose_visu_path']):
        print(" --- Created pose visualization output directory: {} ---".format(params['pose_visu_path']))
        os.makedirs(params['pose_visu_path'])


def draw_2d_box(image, box, color, thickness=2):
    """
        Draws a box on an image with a given color.
        Inputs:
            image     : The image to draw on.
            box       : A list of 4 elements (x1, y1, x2, y2).
            color     : The color of the box.
            thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def show_keypoints(vis, obj_id, im_coords, lcol=(0, 0, 255)):
    """
        Displaying the keypoints of a given object
        Displaying a single point in the input image (vis)
    """

    # for all markers
    cx = int(im_coords[0])
    cy = int(im_coords[1])
    vis = cv2.circle(vis, (cx, cy), 4, lcol, -1, cv2.LINE_AA)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 255)
    cv2.putText(vis, obj_id, (cx, cy + 20), font_face, 0.45, font_color, 1, lineType=cv2.LINE_AA)

    return vis


def draw_box_3d(image, corners, c=(0, 0, 255)):
    """
    Drawing the edges of a 3D cuboid into an image. The frontal face is indicated by crossed lines
    Input:
        image   - 2D image
        corners - back-projected (3D) corner locations, defined as n x 2 image points
        c       - line color
    Output:
        image   - the input image with an edge overlay
    """
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [4, 5, 6, 7]]  # last entry - front face indication
    for ind_f in range(4, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), c, 1, lineType=cv2.LINE_AA)

        # drawing crossing lines for the front face
        if ind_f == 4:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]), (corners[f[2], 0], corners[f[2], 1]), c, 1,
                     lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]), (corners[f[3], 0], corners[f[3], 1]), c, 1,
                     lineType=cv2.LINE_AA)

    return image


def setup_results_dict(k_matrix, img_fname):
    """
        Set the setup parameters for the yaml file to be exported.
        For dynamic parameters initialize to Zeros.
        For parameters that do not change per view set the corresponding values
        This includes the k_matrix, experiment name and file count
    """
    data_dict = edict()
    for cur_obj in bpy.data.collections['POSE_OBJECTS'].all_objects:
        if cur_obj.hide_render:
            continue

        obj_id = cur_obj.name
        data_dict[obj_id] = edict()
        data_dict[obj_id].bb = edict()
        data_dict[obj_id].tdbb = np.zeros((8, 3), dtype=np.float32)
        data_dict[obj_id].dimensions = edict()
        data_dict[obj_id].center = edict()
        data_dict[obj_id].translation = edict()
        data_dict[obj_id].rotation = edict()
        data_dict[obj_id].rotation.absolute = edict()
        data_dict[obj_id].rotation.apparent = edict()
        data_dict[obj_id].bb.ul_x = 0
        data_dict[obj_id].bb.ul_y = 0
        data_dict[obj_id].bb.lr_x = 0
        data_dict[obj_id].bb.lr_y = 0
        data_dict[obj_id].dimensions.x = 0.0
        data_dict[obj_id].dimensions.y = 0.0
        data_dict[obj_id].dimensions.z = 0.0
        data_dict[obj_id].center.x = 0.0
        data_dict[obj_id].center.y = 0.0
        data_dict[obj_id].center.z = 0.0
        data_dict[obj_id].translation.x = 0.0
        data_dict[obj_id].translation.y = 0.0
        data_dict[obj_id].translation.z = 0.0
        data_dict[obj_id].rotation.absolute.x = 0.0
        data_dict[obj_id].rotation.absolute.y = 0.0
        data_dict[obj_id].rotation.absolute.z = 0.0
        data_dict[obj_id].rotation.apparent.x = 0.0
        data_dict[obj_id].rotation.apparent.y = 0.0
        data_dict[obj_id].rotation.apparent.z = 0.0
        # percentage of pixels of the camera image belonging to object - max : image_width * image_height
        data_dict[obj_id].view_pix_perc = 0.0
        # percentage of pixels that are occluded from camera view - max : viewable pixels if no other objects
        # were present
        data_dict[obj_id].occ_pix_perc = 0.0

        data_dict.setup = edict()
        data_dict.setup.cam_mat = k_matrix
        data_dict.setup.img_file = img_fname

    return data_dict


def set_yaml_params(data_dict, obj_id, cx, cy, delta_z, x_app, y_app, z_app, x_abs, y_abs, z_abs,
                    x_dim, y_dim, z_dim, ul_x, ul_y, lr_x, lr_y, tdbb, t_vec, fspy_scale):
    """
        Set the pose parameters for the results yaml
    """
    if fspy_scale == 'cm':
        delta_z /= 100.0
        x_dim /= 100.0
        y_dim /= 100.0
        z_dim /= 100.0
        tdbb /= 100.0

    data_dict[obj_id].center.x = float(cx)
    data_dict[obj_id].center.y = float(cy)
    data_dict[obj_id].center.z = float(delta_z)

    # Swapping XY since in images, x is typically the wide side while in a 3D coordinate frame this is y
    data_dict[obj_id].translation.x = float(t_vec[1])
    data_dict[obj_id].translation.y = float(t_vec[0])
    data_dict[obj_id].translation.z = float(t_vec[2])
    data_dict[obj_id].rotation.apparent.x = float(x_app)
    data_dict[obj_id].rotation.apparent.y = float(y_app)
    data_dict[obj_id].rotation.apparent.z = float(z_app)
    data_dict[obj_id].rotation.absolute.x = float(x_abs)
    data_dict[obj_id].rotation.absolute.y = float(y_abs)
    data_dict[obj_id].rotation.absolute.z = float(z_abs)
    data_dict[obj_id].dimensions.x = float(x_dim)
    data_dict[obj_id].dimensions.y = float(y_dim)
    data_dict[obj_id].dimensions.z = float(z_dim)
    data_dict[obj_id].bb.ul_x = float(ul_x)
    data_dict[obj_id].bb.ul_y = float(ul_y)
    data_dict[obj_id].bb.lr_x = float(lr_x)
    data_dict[obj_id].bb.lr_y = float(lr_y)
    data_dict[obj_id].tdbb = tdbb

    return data_dict


def set_occ_yaml(data_dict, shortest_dist_arr, width, height):
    """
        This method calculates occlusion for all objects using the shortest ray-hit distance array obtained previously
        This is somewhat slow and has potential for optimization
    """

    result_arr = np.full((shortest_dist_arr.shape[0], height, width), 0)
    img_area = int(width * height)
    for y in range(height):
        for x in range(width):
            cur_vec = shortest_dist_arr[:, y, x]
            if cur_vec.min() == 999:
                continue
            hit_idx = np.where(cur_vec == cur_vec.min())
            occ_indices = np.intersect1d(np.where(cur_vec != cur_vec.min()), np.where(cur_vec != 999))
            if hit_idx:
                result_arr[hit_idx, y, x] = 1
            if len(occ_indices) > 0:
                result_arr[occ_indices, y, x] = -1

    obj_idx = 0
    for cur_obj in bpy.data.collections['POSE_OBJECTS'].all_objects:
        if cur_obj.hide_render:
            continue
        obj_id = cur_obj.name
        obj_area = np.count_nonzero(result_arr[obj_idx, :, :] != 0)
        obj_area_viewable = np.count_nonzero(result_arr[obj_idx, :, :] == 1)
        obj_area_occ = np.count_nonzero(result_arr[obj_idx, :, :] == -1)
        data_dict[obj_id].view_pix_perc = obj_area_viewable / max(1, img_area) * 100
        data_dict[obj_id].occ_pix_perc = obj_area_occ / max(1, obj_area) * 100

        obj_idx += 1

    return data_dict


def export_anno_yaml(data_dict, fname):
    """
        Yaml file export function taking an input dict and saving it into a yaml file
        Inputs:
            data_dict: dictionary containing the data fiels of various ground truth annotations or config parameters
            fname: name of the yaml file to dump the data into
    """
    # opening the output file for writing
    cv_file = cv2.FileStorage(fname, cv2.FILE_STORAGE_WRITE)
    fpath = Path(fname).resolve()

    # parsing the individual fields (keys)
    img_name = data_dict.setup.img_file
    cam_mat_float = np.asarray(data_dict.setup.cam_mat, dtype=np.float32)

    cv_file.write("img_file", str(relpath(img_name,fpath.parent).as_posix() ))
    cv_file.write("cam_mat", cam_mat_float)

    for cur_obj in bpy.data.collections['POSE_OBJECTS'].all_objects:
        if cur_obj.hide_render:
            continue

        obj_id = cur_obj.name
        dims_float = np.asarray([v for v in data_dict[obj_id].dimensions.values()], dtype=np.float32)
        bb_int = np.asarray([v for v in data_dict[obj_id].bb.values()], dtype=np.int32)
        tdbb_float = np.asarray(data_dict[obj_id].tdbb, dtype=np.float32)
        center_float = np.asarray([v for v in data_dict[obj_id].center.values()], dtype=np.float32)
        t_float = np.asarray([v for v in data_dict[obj_id].translation.values()], dtype=np.float32)
        app_float = np.asarray([v for v in data_dict[obj_id].rotation.apparent.values()], dtype=np.float32)
        abs_float = np.asarray([v for v in data_dict[obj_id].rotation.absolute.values()], dtype=np.float32)
        view_pix_perc_float = np.asarray(data_dict[obj_id].view_pix_perc, dtype=np.float32)
        occ_pix_perc_float = np.asarray(data_dict[obj_id].occ_pix_perc, dtype=np.float32)
        cv_file.write(f'{obj_id}_dims', dims_float)
        cv_file.write(f'{obj_id}_bb2d', bb_int)
        cv_file.write(f'{obj_id}_bb3d', tdbb_float)
        cv_file.write(f'{obj_id}_center', center_float)
        cv_file.write(f'{obj_id}_translation', t_float)
        cv_file.write(f'{obj_id}_rotation_app', app_float)
        cv_file.write(f'{obj_id}_rotation_abs', abs_float)
        cv_file.write(f'{obj_id}_view_pix_perc', view_pix_perc_float)
        cv_file.write(f'{obj_id}_occ_pix_perc', occ_pix_perc_float)

    cv_file.release()


def export_config_yaml(data_dict, fname):
    """
        Export config parameters to a yaml
    """
    cv_file = cv2.FileStorage(fname, cv2.FILE_STORAGE_WRITE)

    # parsing the individual fields (keys)
    for k, v in data_dict.items():
        cv_file.write(k, str(v))

    cv_file.release()


def save_result_im(img, fname, visu_path):
    """
        Image export (save as png) either as the rendered (result) image (in case of debugMode = False) or as a debugdr
        image, containing visualizations for visual control of correctness
        Inputs:
            img:       rendered image (debugMode = False) or rendered image containing debug visualizations
            (debugMode = True)
            icnt:      running index [0..N]
            debugMode: flag selecting the nature of exported image. In case of result render, image is saved into the
            img_path folder.
                       in the debugMode = True case, image is saved into the dbg_path folder.
        Output:
            fname_full: returns the full name (without path) of the saved image
    """

    fname_full = "visu_" + fname
    out_path = os.path.join(visu_path, fname_full)

    cv2.imwrite(out_path, img)
