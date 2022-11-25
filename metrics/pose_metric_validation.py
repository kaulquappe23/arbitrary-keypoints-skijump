
import numpy as np

from data.skijump.skijump_joint_order import yt_skijump_reference_joints
from metrics.pose_metrics import pck_normalized_distances_fast, pck_scores_from_normalized_distances, pck_score_at_threshold
from utils.general_utils import get_dict


def eval_pck(preds, annos, cfg, pck_config=None, val=False):
    events = []
    assert len(preds) == len(annos)

    num_joints = cfg.NUM_JOINTS
    use_anno = [i for i in range(num_joints)]

    pck_thresholds = get_dict(pck_config, "pck_thresholds", [0.1])
    annotations = np.zeros((len(preds), num_joints, 3))
    predictions = np.zeros((len(preds), num_joints, 3))

    for i in range(len(preds)):
        if annos is not None:
            annotations[i] = annos[i][0, 0, use_anno, :]
        if len(preds[i]) > 0:
            predictions[i] = preds[i][0][use_anno, :3]

    annotations_full = None
    predictions_full = None
    if getattr(cfg, "GENERATE_KEYPOINTS_SEGMENTATION", None) is not None and cfg.GENERATE_KEYPOINTS_SEGMENTATION[1] > 0:
        add_joints = cfg.GENERATE_KEYPOINTS_SEGMENTATION[1]
        if val and hasattr(cfg, "GENERATE_KEYPOINTS_SEGMENTATION_VAL"):
            add_joints = cfg.GENERATE_KEYPOINTS_SEGMENTATION_VAL
        n_joints = cfg.NUM_JOINTS + add_joints
        predictions_full = np.zeros((len(preds), n_joints, 3))
        for i in range(len(preds)):
            predictions_full[i] = preds[i][0]
        if annos is not None:
            annotations_full = np.asarray(annos)[:, 0, 0]

    pck_full = None
    separate_ref_lengths = None

    if cfg.NAME == "yt_skijump":
        fallback_reference_lengths = None
        reference_joints = yt_skijump_reference_joints
    else:
        raise RuntimeError("PCK information for dataset missing")

    pck, joint_pck = eval_pck_array(predictions, annotations, events, fallback_reference_lengths, reference_joints, pck_thresholds, None)

    if annotations_full is not None:
        pck_full, joint_pck_full = eval_pck_array(predictions_full, annotations_full, events, fallback_reference_lengths, reference_joints, pck_thresholds, None, separate_ref_lengths=separate_ref_lengths)

    if pck_config is not None and pck_config['joint_wise']:
        pck = (pck, joint_pck)
        if pck_full is not None:
            pck_full = (pck_full, joint_pck_full)

    return pck, pck_full


def eval_pck_array(prediction_array, annotation_array, events, fallback_ref_length_func, reference_joints, pck_thresholds, use_indices, separate_ref_lengths=None):
    """
    Evaluate predictions according to pck metric
    @param prediction_array: array containing predictions
    @param annotation_array: array containing annotations, in same order
    @param reference_joints: these joints are used in order to calculate the reference length (normally the reference length is the torso size, hence shoulder and hip are usual reference joints)
    @param fallback_ref_length_func: if the desired joints are not annotated, reference lengths can be used. This is a FUNCTION that is called then. Can return None if this behavior is not needed
    @param events: used for fallback_ref_length_func, can contain information how to compute the reference lengths
    @param pck_thresholds: PCK values for these thresholds are returned (typical values are 0.1, 0.2)
    @param use_indices: the PCK values are calculated only based on these joint indices. Set to None if you want to use all joints
    @return: list of total PCK scores for the given thresholds and list of PCK scores for each joint
    """
    # In order to calculate PCK metrics, the left shoulder and right hip have to be annotated
    # If this is not the case, we have to provide a fallback reference length to normalize keypoint errors
    if fallback_ref_length_func is None:
        fallback_ref_lengths = None
    else:
        fallback_ref_lengths = fallback_ref_length_func(events, annotation_array)

    # Get normalized PCK distances
    normalized_distances = pck_normalized_distances_fast(prediction_array, annotation_array,
                                                         ref_length_indices=reference_joints,
                                                         fallback_ref_lengths=fallback_ref_lengths,
                                                         separate_ref_lengths=separate_ref_lengths)
    # Calculate PCK scores
    pck_ts, pck_scores = pck_scores_from_normalized_distances(normalized_distances, ignore_negative_distances=True, relevant_indices=use_indices)

    # Collect PCK scores at fixed distance thresholds
    print_scores = list()
    for pck_threshold in pck_thresholds:
        print_scores.append(pck_score_at_threshold(pck_ts, pck_scores, applied_threshold=pck_threshold))

    joint_scores = []
    for joint in range(annotation_array.shape[1]):
        # Calculate PCK scores
        pck_ts, pck_scores = pck_scores_from_normalized_distances(normalized_distances, ignore_negative_distances=True, relevant_indices=[joint])
        # Collect PCK scores at fixed distance thresholds
        scores = []
        for pck_threshold in pck_thresholds:
            scores.append(pck_score_at_threshold(pck_ts, pck_scores, applied_threshold=pck_threshold))
        joint_scores.append(scores)

    return print_scores, joint_scores