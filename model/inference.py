
import copy

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data.build import make_test_dataloader
from data.transform_utils import flip_back, affine_transform, get_affine_transform
from data.transformer_dataset import fliplr_keypoint_vectors
from model.transformer import token_pose


def inference(config, data_loader, dataset, model, use_flip=True, val=False, return_orig_annos=False):
    # switch to evaluate mode
    model.eval()

    num_samples = len(dataset)
    if val and hasattr(config, "GENERATE_KEYPOINTS_SEGMENTATION_VAL"):
        additional_joints = (config.GENERATE_KEYPOINTS_SEGMENTATION_VAL, config.GENERATE_KEYPOINTS_SEGMENTATION_VAL)
    else:
        additional_joints = getattr(config, "GENERATE_KEYPOINTS_SEGMENTATION", (0, 0))
    num_joints = config.NUM_JOINTS
    num_joints += additional_joints[1]

    all_preds = np.zeros(
        (num_samples, num_joints, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    idx = 0
    joints = []
    keypoint_vectors = []
    thickness_vectors = []
    original_annotations = []

    with torch.no_grad():
        for i, (input, target, target_weight, meta) in tqdm(enumerate(data_loader), position=0, leave=True, desc="Inference", total=num_samples):
            # compute output

            keypoint_vec = meta['keypoint_vectors'].cuda()
            thickness_vec = meta['thickness_vectors'].cuda()
            orig_annos = meta['non_transformed'].numpy()
            output = model(input.cuda(), keypoint_vector=keypoint_vec, thickness_vector=thickness_vec)

            keypoint_vectors.append(copy.deepcopy(keypoint_vec.cpu().numpy()[0]))
            thickness_vectors.append(copy.deepcopy(thickness_vec.cpu().numpy()[0]))
            original_annotations.append(orig_annos)

            if use_flip:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                temp_vec = fliplr_keypoint_vectors(keypoint_vec.cpu().numpy()[0], config.FLIP_PAIRS)
                keypoint_vec = torch.from_numpy(temp_vec[None, :, :]).cuda()
                thickness_vec[:, :, [0, 2]] = thickness_vec[:, :, [2, 0]]
                outputs_flipped = model(input_flipped, keypoint_vector=keypoint_vec, thickness_vector=thickness_vec)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = output_flipped.cpu().numpy()
                output_flipped = flip_back(output_flipped, [])

                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            num_images = input.size(0)

            c = meta['center'].numpy()
            s = meta['scale'].numpy()

            preds, maxvals = get_final_preds(output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :num_joints, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals[:, :num_joints]
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = 1
            image_id = meta['image_id']
            image_path.extend(image_id)
            joints.append(meta["joints"].numpy())

            idx += num_images

        return all_preds, all_boxes, image_path, (keypoint_vectors, original_annotations)


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
        dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
        dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
        dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
        dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
            + hm[py-1][px-1])
        dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
        derivative = np.matrix([[dx],[dy]])
        hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i,j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border: -border, border: -border] = hm[i,j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i,j] = dr[border: -border, border: -border].copy()
            hm[i,j] *= origin_max / np.max(hm[i,j])
    return hm


def get_final_preds(hm, center, scale, rot=0, flip_width=None):
    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]

    # post-processing
    if True:  # config.TEST.POST_PROCESS:
        hm = gaussian_blur(hm, 11)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n,p] = taylor(hm[n][p], coords[n][p])

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height], 0 if isinstance(rot, int) and rot == 0 else rot[i]
        )
        if flip_width is not None:
            preds[i, :, 0] = flip_width[i] - preds[i, :, 0] - 1

    return preds, maxvals


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords



def build_inference_datasets(cfg, pretrained_file, subset=None, verbose=True):

    model = token_pose.get_pose_net(cfg, verbose=verbose)
    model.init_weights(pretrained_transformer=pretrained_file, verbose=verbose)
    model = model.cuda()

    subset = cfg.TEST_SUBSET if subset is None else subset
    data_loader, dataset = make_test_dataloader(cfg, subset=subset, verbose=verbose)

    return model, data_loader, dataset

