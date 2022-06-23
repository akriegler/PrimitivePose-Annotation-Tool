# ----------------------------------------------------------------------------------------------------------
#    FUNCTION: get_pose_anno.py
# ----------------------------------------------------------------------------------------------------------
#    DESCRIPTION:
#    Blender script for obtaining and exporting pose annotations for arbitrary objects
#    Acknowledgement: Several camera functions by Markus Murschitz are used. His support is greatly
#                     acknowledged
# ----------------------------------------------------------------------------------------------------------
#    RESPONSIBILITY: A. Kriegler
#    VERSIONs:       python 3.9
#    CREATION DATE:  2022/02/02
# ----------------------------------------------------------------------------------------------------------
#    INPUT: Blender objects
#    OUTPUT: Pose annotations
#    DEPENDENCIES: bpy, cv2, numpy
# ----------------------------------------------------------------------------------------------------------
#    AIT - Austrian Institute of Technology
# ----------------------------------------------------------------------------------------------------------
#    REVISION CONTROL: See git logs
# ----------------------------------------------------------------------------------------------------------

# Blender python packages (come with Blender installation)
import bpy  # This is the blender python package and exists within Blender's own python installation

# Default packages
import os
import sys

# Additional packages
import datetime

import cv2
import numpy as np
from pathlib import Path

root_path = "path-to-repo"

utils_path = root_path + '/utils'

if utils_path not in sys.path:
    sys.path.append(utils_path)

if root_path not in sys.path:
    sys.path.append(root_path)

# Custom packages
import cam_tools as ctools
import pose_tools as ptools
import results_tools as rtools

import importlib

importlib.reload(ctools)
importlib.reload(ptools)
importlib.reload(rtools)

# initializing the parameter dictionary
params = {}


def set_params(camera_name, img_name):
    global params

    # result configs
    params['export_yaml'] = True
    params['get_occlusion'] = True
    params['save_blend_file'] = True

    # path configs
    img_abspath = Path(bpy.path.abspath(bpy.data.images[img_name].filepath))

    params['out_path'] = img_abspath.parent.parent.resolve()  # Path('D:/xpose_stuff/data/clutterd_primitives/view_0/large_obj/pose')
    gt_path = params['out_path'] / 'gt'
    params['pose_gt_path'] = gt_path / 'anno_files'
    params['pose_visu_path'] = gt_path / 'debug_imgs'
    params['pose_blend_path'] = gt_path / 'blend_files'
    params['img_name'] = img_abspath.name  # name of the image file for pose annotation
    params['img_path'] = img_abspath.resolve()

    # other configs
    params['res_x'], params['res_y'] = bpy.data.images[img_name].size
    params['force_symmetry_angle_zero'] = True
    params['force_planar_angle_zero'] = True
    params['fspy_scale'] = 'm'


# --------------------------------
# FUNCTION run - main code entry
# --------------------------------
# - main entry --
if __name__ == "__main__":

    camera_objs = [cam for cam in bpy.data.objects if (cam.type == 'CAMERA')]
    for c in camera_objs:
        if c.hide_render:
            continue

        assert len(c.data.background_images) == 1, "a camera is expected to have only one background image asociated with it"

        img_name = c.data.background_images[0].image.name
        set_params(c.name, img_name)

        # --------------------------------------------------------------------------------------------------------
        # clearing the console (both windows and linux)
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")

        # generating the directory structure
        rtools.generate_dir_structure(params)

        bpy.ops.outliner.orphans_purge()

        # compositing flag must be enabled
        bpy.context.scene.render.use_compositing = True
        env_scene = bpy.data.scenes['Scene']
        env_world = env_scene.world

        cam = bpy.data.objects.get(c.name, None)
        bpy.context.scene.camera = cam
        cam_data = cam.data
        bpy.data.scenes["Scene"].render.use_multiview = False

        # retrieving the camera matrix
        k = np.array(ctools.get_blender_k_matrix(cam.data))
        #print("\n Camera matrix: \n{}\n".format(k))

        # relevant intrinsics
        fx = k[0, 0]  # focal length x
        fy = k[1, 1]  # focal length y
        px = k[0, 2]  # principal point x
        py = k[1, 2]  # principal point y

        center_cxy = [0, 0, 0]
        ul_x, ul_y, lr_x, lr_y, xdim, ydim, zdim, t_vec = [0 for _ in range(8)]
        x_t, y_t, z_t, x_abs, y_abs, z_abs = [0 for _ in range(6)]
        x_app, y_app, z_app, delta_z = [0 for _ in range(4)]

        # number of pixels in X/Y direction
        width = int(params['res_x'] * (bpy.context.scene.render.resolution_percentage / 100))
        height = int(params['res_y'] * (bpy.context.scene.render.resolution_percentage / 100))
        # get vectors which define view frustum of camera
        frame = cam.data.view_frame(scene=bpy.context.scene)
        u_r = frame[0]
        b_l = frame[2]
        u_l = frame[3]

        # setup vectors to match pixels
        x_range = np.linspace(u_l[0], u_r[0], width)
        y_range = np.linspace(u_l[1], b_l[1], height)
        img_area = width * height

        num_objs = 0

        for obj_idx, cur_obj in enumerate(bpy.data.collections["POSE_OBJECTS"].all_objects):
            if not cur_obj.hide_render:
                num_objs += 1

        max_number_objs = num_objs
        shortest_dist_arr = np.full((max_number_objs, y_range.size, x_range.size), 999, dtype=np.float32)

        result_dict = rtools.setup_results_dict(k, params['img_path'])

        # create random config of primitiva and obtain gt annotations
        file_cnt = 0
        obj_idx = 0
        rl_script_start = datetime.datetime.now()

        vis_im = cv2.imread(str(params['img_path']))[..., :3]

        for cur_obj in bpy.data.collections["POSE_OBJECTS"].all_objects:
            # pose estimation is only computed for rendered (render flag set to true objects)
            if cur_obj.hide_render:
                continue

            obj_id = cur_obj.name

            bpy.context.view_layer.update()

            # main pose parameter measurement step
            pose_params = ptools.get_6dof_pose_params(cam, obj_id, cur_obj, fx, fy, px, py)

            # parsing variables
            center_cxy = pose_params[0:2]
            ul_x, ul_y, lr_x, lr_y = pose_params[2], pose_params[3], pose_params[4], pose_params[5]
            xdim, ydim, zdim = pose_params[6], pose_params[7], pose_params[8]
            t_vec = pose_params[9]
            delta_z = pose_params[10]

            # this is the key function to determine the relative orientations
            x_app, y_app, z_app = ptools.get_rel_orient_to_camera(cam, cur_obj.data.name, cur_obj, fx, fy, px, py,
                                                                  params['res_x'],
                                                                  params['force_symmetry_angle_zero'],
                                                                  params['force_planar_angle_zero'])

            x_t, y_t, z_t, x_abs, y_abs, z_abs = ptools.decode_orientations(x_app, y_app, z_app, center_cxy, fx, fy,
                                                                            px, py,
                                                                            params['res_x'])

            shortest_dist_arr, ul_x, ul_y, lr_x, lr_y = ptools.get_shortest_dist(cur_obj, cam, shortest_dist_arr,
                                                                                 x_range, y_range,
                                                                                 [ul_x, ul_y, lr_x, lr_y], obj_idx,
                                                                                 cam.data.view_frame(
                                                                                    scene=bpy.context.scene)[3])

            # DISPLAY: show back-projected object center
            vis_im = rtools.show_keypoints(vis_im, obj_id, center_cxy, (255, 0, 0))
            # DISPLAY: drawing the 2D bounding box [ul_x, ul_y, lr_x, lr_y]
            rtools.draw_2d_box(vis_im, [ul_x, ul_y, lr_x, lr_y], (0, 255, 0))

            # re-projection based on the pose parameters
            corners_3d = ptools.compute_box_3d([xdim, ydim, zdim], t_vec, x_t, y_t, z_t)
            box_2d = ptools.project_to_image(corners_3d, k)
            vis_im = rtools.draw_box_3d(vis_im, box_2d.astype(int))
            # vis_im = rtools.insert_info_into_image(vis_im, [xdim, ydim, zdim], [x_abs, y_abs, z_abs],
            #                                       [x_app, y_app, z_app], t_vec)
            if 'Cylinder' in cur_obj.data.name:
                y_app = 0.0

            result_dict = rtools.set_yaml_params(result_dict, obj_id, center_cxy[0], center_cxy[1], delta_z, x_app,
                                                 y_app,
                                                 z_app, x_abs, y_abs, z_abs, xdim, ydim, zdim, ul_x, ul_y, lr_x,
                                                 lr_y,
                                                 corners_3d, t_vec, params['fspy_scale'])

            obj_idx += 1

        out_fname = os.path.join(params['pose_gt_path'], '{}_gt.yml'.format(params['img_name'].split('.')[0]))

        if params['get_occlusion']:
            result_dict = rtools.set_occ_yaml(result_dict, shortest_dist_arr, width, height)

        rtools.export_anno_yaml(result_dict, out_fname)

        # Debug visualization export
        rtools.save_result_im(vis_im, params['img_name'],
                              params['pose_visu_path'])  # Debug is True for debug overlays

        if params['save_blend_file']:
            bpy.ops.wm.save_as_mainfile(filepath=str(params['pose_blend_path']) + '/' + str(params['img_name']).split('.')[0] + '.blend' )

    print('====== Annotation export finished =====\n')
