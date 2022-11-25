
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.config_utils import get_transformer_config_by_name

import pickle

import numpy as np

from data.skijump.skijump_joint_order import YTSkijumpJointOrder
from data.skijump.yt_skijump_data import YTSkijump
from metrics.pose_metric_validation import eval_pck
from metrics.thickness_metric import format_thickness_result, thickness_metric, thickness_percentage_differences
from paths import YTSkijumpLoc
from model.inference import build_inference_datasets, inference
from utils.general_utils import format_result, gpu_settings
from utils.training_utils import set_deterministic


def general_eval_pipeline(execute_again, weights_file, dump_inference_results_file):
    """
    Evaluate a model trained for arbitrary keypoint generation for the youtube skijump dataset
    @param execute_again: if True, the inference is executed again. If false, the dump_inference_results_file has to exist and will be used for metric calculation
    @param weights_file: model weights file
    @param dump_inference_results_file: temporary file to dump the inference results
    """

    set_deterministic()
    gpu_settings()

    config  = get_transformer_config_by_name("yt_skijump-seg")
    config.TEST_SUBSET = "test"
    config.WORKERS = 16
    config.GENERATE_KEYPOINTS_SEGMENTATION_VAL = 200

    config2 = get_transformer_config_by_name("yt_skijump", vector_encoded=True)
    config2.TEST_SUBSET = "test"
    config2.WORKERS = 16

    if execute_again:
        model, data_loader, dataset = build_inference_datasets(config2, weights_file, verbose=True)
        res2 = inference(config2, data_loader, dataset, model, val=True)
        model, data_loader, dataset = build_inference_datasets(config, weights_file, verbose=True)
        res = inference(config, data_loader, dataset, model, val=True)
        with open(dump_inference_results_file, "wb") as f:
            pickle.dump((res, res2), f)
    with open(dump_inference_results_file, "rb") as f:
        res, res2 = pickle.load(f)

        print("--- results on segmentation data ---")

        all_preds, all_boxes, ids, joints = res
        orig_annos = joints[-1]
        thick_vec = None if len(joints) == 2 else joints[1]
        preds = [all_preds[idx][None, :, :] for idx in range(all_preds.shape[0])]
        annos = [orig_annos[idx][None, :, :] for idx in range(len(orig_annos))]

        metric_res = eval_pck(preds, annos, config, None, val=True)
        print(format_result(metric_res, header="", names=YTSkijumpJointOrder.names()))

        parameters = {
            "dataset_dir": os.path.join(YTSkijumpLoc.base_path, YTSkijumpLoc.annotation_path),
            "subset": "test",
            "bbox": True,
            "bodypart_segmentations": os.path.join(YTSkijumpLoc.base_path, YTSkijumpLoc.segmentation_path),
            "verbose": False
        }
        yt_skijump = YTSkijump(parameters)

        orig_annos = np.asarray(orig_annos)[:, 0, :]
        thickness_res = thickness_percentage_differences(all_preds, orig_annos, yt_skijump.get_num_keypoints(), yt_skijump.bodypart_dict, yt_skijump.load_bodypart_mask, None, ids, thick_vec,
                                                         bodyparts_no_center=yt_skijump.bodypart_ids_no_center)

        mean, std, pct = thickness_metric(thickness_res[0])
        format_thickness_result(mean, std, pct, "")

        print("--- results on all data ---")

        all_preds, all_boxes, ids, joints = res2
        orig_annos = joints[-1]
        preds = [all_preds[idx][None, :, :] for idx in range(all_preds.shape[0])]
        annos = [orig_annos[idx][None, :, :] for idx in range(len(orig_annos))]

        metric_res = eval_pck(preds, annos, config2, None, val=True)
        print(format_result(metric_res, header="", names=YTSkijumpJointOrder.names()))

    print("---------------------------------------------------------------\n\n")


if __name__ == '__main__':

    weights_file = YTSkijumpLoc.weights
    dump_inference_results = YTSkijumpLoc.dump_inference_results
    print("Evaluating {}".format(weights_file))
    step = weights_file[weights_file.rfind("/") + 1: -8]
    run = weights_file[weights_file.rfind("20"): weights_file.rfind("/")]
    general_eval_pipeline(True, weights_file, dump_inference_results)
