import sys
sys.path.append('../../../camera-control/')

import math
import os
import cv2
import torch
import numpy as np

from argparse import Namespace
from typing import Dict, List, Optional, Any, Tuple

from camera_control.Module.camera import Camera

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from models.diffusion import CasDiffMVS


DEFAULT_CASDIFFMVS_ARGS = dict(
    numdepth_initial=48,
    numdepth=384,
    scale=[0.0, 0.125, 0.025],
    sampling_timesteps=[0, 1, 1],
    ddim_eta=[0, 1, 1],
    timesteps=[1000, 1000, 1000],
    stage_iters=[1, 3, 3],
    cost_dim_stage=[4, 4, 4],
    CostNum=[0, 4, 4],
    hidden_dim=[0, 32, 20],
    context_dim=[32, 32, 16],
    unet_dim=[0, 16, 8],
    min_radius=0.125,
    max_radius=8,
)


class Detector(object):
    def __init__(
        self,
        model_file_path: Optional[str] = None,
        device: str = 'cuda:0',
        model_args: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.model: Optional[CasDiffMVS] = None
        self.args: Optional[Namespace] = None

        merged = dict(DEFAULT_CASDIFFMVS_ARGS)
        if model_args is not None:
            merged.update(model_args)
        self.args = Namespace(**merged)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(
        self,
        model_file_path: str,
    ) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Detector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            return False

        self.model = CasDiffMVS(self.args, test=True)

        state_dict = torch.load(model_file_path, map_location='cpu')
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        print('[INFO][Detector::loadModel]')
        print('\t model loaded from:', model_file_path)
        return True

    @staticmethod
    def _prepareImages(
        camera_list: List[Camera],
        base: int = 32,
        max_h: int = 4800,
        max_w: int = 6400,
    ) -> Tuple[List[np.ndarray], int, int, int, int]:
        """Convert Camera images to model-ready tensors with bottom/right padding.

        Images are optionally down-scaled when exceeding *max_h* / *max_w* and
        then padded (bottom & right) to the next multiple of *base*.  Because
        padding is only appended (never prepended), the camera principal point
        stays at the same pixel coordinate and the intrinsic matrix does not
        need an extra translation adjustment.

        Returns:
            imgs_np: list of V numpy arrays, each (3, pad_h, pad_w), float32 [0,1]
            orig_h:  height before padding (after optional down-scale)
            orig_w:  width  before padding (after optional down-scale)
            pad_h:   padded height  (multiple of *base*)
            pad_w:   padded width   (multiple of *base*)
        """
        ref_cam = camera_list[0]
        h, w = ref_cam.height, ref_cam.width

        need_downscale = h > max_h or w > max_w
        if need_downscale:
            scale = min(max_w / w, max_h / h)
            orig_w = int(w * scale)
            orig_h = int(h * scale)
        else:
            orig_w, orig_h = w, h

        pad_w = int(math.ceil(orig_w / base) * base)
        pad_h = int(math.ceil(orig_h / base) * base)

        imgs_np = []
        for cam in camera_list:
            img = cam.image
            if img is None:
                raise ValueError(
                    f'Camera {getattr(cam, "image_id", "?")} has no image loaded'
                )
            img_np = img.detach().cpu().float().numpy()
            if img_np.max() > 1.5:
                img_np = img_np / 255.0

            if img_np.shape[0] != orig_h or img_np.shape[1] != orig_w:
                img_np = cv2.resize(img_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            if orig_h != pad_h or orig_w != pad_w:
                padded = np.zeros((pad_h, pad_w, img_np.shape[2]), dtype=np.float32)
                padded[:orig_h, :orig_w, :] = img_np
                img_np = padded

            imgs_np.append(img_np.transpose(2, 0, 1).astype(np.float32))

        return imgs_np, orig_h, orig_w, pad_h, pad_w

    @staticmethod
    def _buildProjMatrices(
        camera_list: List[Camera],
        orig_h: int,
        orig_w: int,
    ) -> Dict[str, np.ndarray]:
        """Build multi-stage projection matrices from Camera list.

        *orig_h* / *orig_w* are the image dimensions **after** optional
        down-scaling but **before** padding.  The intrinsic is scaled
        accordingly so that pixel coordinates match the actual image content.

        The extrinsic uses world2cameraColmap (COLMAP convention: X-right,
        Y-down, Z-forward) which matches the MVS cam.txt format used in
        training.
        """
        ref_cam = camera_list[0]
        scale_w = orig_w / ref_cam.width
        scale_h = orig_h / ref_cam.height

        proj_list = []
        for cam in camera_list:
            extrinsic = cam.world2cameraColmap.detach().cpu().float().numpy()

            intrinsic = cam.intrinsic.detach().cpu().float().numpy()
            intrinsic[0, :] *= scale_w
            intrinsic[1, :] *= scale_h

            proj_mat = np.zeros((2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsic
            proj_mat[1, :3, :3] = intrinsic
            proj_list.append(proj_mat)

        proj_matrices = np.stack(proj_list)

        stage1 = proj_matrices.copy()
        stage1[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125
        stage2 = proj_matrices.copy()
        stage2[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage3 = proj_matrices.copy()
        stage3[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5

        return {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": proj_matrices,
        }

    @staticmethod
    def _buildDepthValues(
        depth_min: float,
        depth_max: float,
        numdepth: int,
    ) -> np.ndarray:
        disp_min = 1.0 / depth_max
        disp_max = 1.0 / depth_min
        return np.linspace(disp_min, disp_max, numdepth, dtype=np.float32)

    @staticmethod
    def _depthToPoints(
        depth: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        image: Optional[np.ndarray] = None,
        conf: Optional[np.ndarray] = None,
        conf_threshold: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """Back-project a depth map into a world-space point cloud.

        Args:
            depth: (H, W) depth map
            intrinsic: (3, 3) camera intrinsic (at the depth map resolution)
            extrinsic: (4, 4) world-to-camera (COLMAP convention)
            image: optional (H, W, 3) or (3, H, W) RGB for coloring
            conf: optional (H, W) confidence for masking
            conf_threshold: points below this confidence are discarded

        Returns:
            dict with 'points' (N, 3) and optionally 'colors' (N, 3)
        """
        h, w = depth.shape[:2]
        mask = depth > 0

        if conf is not None and conf_threshold > 0:
            mask = mask & (conf > conf_threshold)

        y, x = np.where(mask)
        z = depth[mask]

        pts_cam = np.linalg.inv(intrinsic) @ np.vstack([x, y, np.ones_like(x)]) * z
        pts_world = (np.linalg.inv(extrinsic) @ np.vstack([pts_cam, np.ones((1, len(z)))]))[:3]

        result: Dict[str, np.ndarray] = {'points': pts_world.T.astype(np.float32)}

        if image is not None and len(z) > 0:
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = image.transpose(1, 2, 0)
            colors = image[mask]
            if colors.size > 0 and colors.max() > 1.5:
                colors = colors / 255.0
            result['colors'] = colors.astype(np.float32)

        return result

    def detectCameras(
        self,
        camera_list: List[Camera],
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        ref_index: int = 0,
        src_indices: Optional[List[int]] = None,
        conf_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Run MVS depth estimation on a set of posed cameras.

        Args:
            camera_list: cameras with images and poses loaded
            depth_min: minimum scene depth (explicit)
            depth_max: maximum scene depth (explicit)
            ref_index: index of the reference view in camera_list
            src_indices: indices of source views; defaults to all others
            conf_threshold: photometric confidence threshold for point cloud

        Returns:
            dict with keys:
                'depth': (H, W) final depth map (numpy)
                'photometric_confidence': list of confidence maps (numpy)
                'points': (N, 3) world-space point cloud
                'colors': (N, 3) per-point RGB (if images available)
                'ref_camera': the reference Camera object
        """
        if self.model is None:
            print('[ERROR][Detector::detectCameras]')
            print('\t model not loaded!')
            return {}

        if src_indices is None:
            src_indices = [i for i in range(len(camera_list)) if i != ref_index]

        ordered_cams = [camera_list[ref_index]] + [camera_list[i] for i in src_indices]

        imgs_np, orig_h, orig_w, pad_h, pad_w = self._prepareImages(ordered_cams)
        proj_matrices_ms = self._buildProjMatrices(ordered_cams, orig_h, orig_w)
        depth_values = self._buildDepthValues(depth_min, depth_max, self.args.numdepth)

        imgs_cuda = [
            torch.from_numpy(img).unsqueeze(0).to(self.device)
            for img in imgs_np
        ]
        proj_ms_cuda = {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in proj_matrices_ms.items()
        }
        depth_values_cuda = torch.from_numpy(depth_values).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(imgs_cuda, proj_ms_cuda, depth_values_cuda)

        final_depth = outputs['depth'][-1][0].cpu().numpy()[:orig_h, :orig_w]
        confidences = [
            c[0].cpu().numpy()[:orig_h, :orig_w]
            for c in outputs['photometric_confidence']
        ]

        ref_cam = ordered_cams[0]
        ref_extrinsic = proj_matrices_ms['stage4'][0, 0, :4, :4]
        ref_intrinsic = proj_matrices_ms['stage4'][0, 1, :3, :3]

        ref_img_np = imgs_np[0][:, :orig_h, :orig_w].transpose(1, 2, 0)

        avg_conf = None
        if len(confidences) > 0:
            avg_conf = confidences[-1]

        pcd = self._depthToPoints(
            final_depth, ref_intrinsic, ref_extrinsic,
            image=ref_img_np,
            conf=avg_conf,
            conf_threshold=conf_threshold,
        )

        return {
            'depth': final_depth,
            'photometric_confidence': confidences,
            'points': pcd['points'],
            'colors': pcd.get('colors'),
            'ref_camera': ref_cam,
        }
