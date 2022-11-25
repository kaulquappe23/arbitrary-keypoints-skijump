
class YTSkijumpLoc:
    # This is the path to the downloaded folder. It contains subfolders with keypoints and segmentations
    base_path = "path/to/published_version_youtube_dataset"
    # Path to the frames extracted from the videos. We expect that this folder has subfolders for each video (named by its index)
    # and each subfolder contains the frames with the following naming convention: <video_id>_(<frame_num>).jpg,
    # whereby the frame num is five-digit, e.g. <frames_path>/0/0_(01944).jpg
    frames_path = "/path/to/annotated_frames"
    annotation_path = "keypoints"
    segmentation_path = "segmentations"
    # This is the path to the model weights file. It is located in this repository in model/pretrained/model_seg_proj.pth.tar
    weights = "path/to/model_seg_proj.pth.tar"
    # Specify a file to dump the inference results
    dump_inference_results = "path/to/dump_inference.pkl"