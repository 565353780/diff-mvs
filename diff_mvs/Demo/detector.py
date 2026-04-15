import sys
sys.path.append('../camera-control/')

import os
import cv2
import numpy as np
import open3d as o3d

from camera_control.Module.camera_convertor import CameraConvertor

from diff_mvs.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = f'{home}/chLi/Model/DiffMVS/casdiffmvs_blendmvg.ckpt'

    colmap_data_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/colmap_normalized/'
    depth_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/depth/'
    depth_vis_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/depth_vis/'
    masked_depth_vis_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/masked_depth_vis/'
    pcd_folder_path = f'{home}/chLi/Dataset/GS/haizei_zihan/pcd/'

    depth_min = 0.1
    depth_max = 10.0

    camera_list = CameraConvertor.loadColmapDataFolder(colmap_data_folder_path)

    print(f'[INFO][demo] loaded {len(camera_list)} cameras')

    detector = Detector(model_file_path)

    os.makedirs(depth_folder_path, exist_ok=True)
    os.makedirs(depth_vis_folder_path, exist_ok=True)
    os.makedirs(masked_depth_vis_folder_path, exist_ok=True)
    os.makedirs(pcd_folder_path, exist_ok=True)

    num_views = min(len(camera_list), 10)
    for ref_idx in range(len(camera_list)):
        if ref_idx == 0:
            src_idx = 1
        else:
            src_idx = ref_idx - 1

        src_indices = [src_idx]
        for j in range(len(camera_list)):
            if j != ref_idx and j != src_idx:
                src_indices.append(j)
        src_indices = src_indices[:num_views - 1]

        result = detector.detectCameras(
            camera_list,
            depth_min=depth_min,
            depth_max=depth_max,
            ref_index=ref_idx,
            src_indices=src_indices,
            conf_threshold=0.3,
        )

        if not result:
            print(f'[WARN][demo] inference failed for camera {ref_idx}')
            continue

        camera = camera_list[ref_idx]
        depth = result['depth']
        cam_h, cam_w = camera.height, camera.width
        if depth.shape[0] != cam_h or depth.shape[1] != cam_w:
            depth = cv2.resize(depth, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)
        camera.loadDepth(depth)

        image_filename = camera.image_id
        fmt = '.' + image_filename.split('.')[-1]
        image_basename = image_filename.split(fmt)[0]

        np.save(depth_folder_path + image_basename + '.npy', camera.depth_with_conf.cpu().numpy())
        cv2.imwrite(depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=False))
        cv2.imwrite(masked_depth_vis_folder_path + image_filename, camera.toDepthVisCV(use_mask=True))

        pts = camera.toDepthPoints(use_mask=True)[0].cpu().numpy().reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(pcd_folder_path + image_basename + '.ply', pcd)

        print(f'[INFO][demo] camera {ref_idx}/{len(camera_list)}: '
              f'depth shape={tuple(camera.depth.shape)}, points={len(pts)}')

    return True
