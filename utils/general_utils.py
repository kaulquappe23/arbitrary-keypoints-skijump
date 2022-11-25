
import os

import cv2
import torch


def get_dict(dic_obj, key, default=None):
    if dic_obj is not None and key in dic_obj and dic_obj[key] is not None:
        return dic_obj[key]
    return default


def gpu_settings():
    cv2.setNumThreads(0)
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.system("ulimit -n 100000")


def format_result(result, header="PCK: ", names=None):
    text = ""
    pck, pck_full = result
    if len(pck) == 1:
        text = header
        text += "{:.1f}".format(round(pck[0]*100, 1))
    else:
        if names is not None:
            text += "{:15}".format("")
            for name in names:
                text += "{} & ".format(name)
            text += "avg \\\\ \n{:15}".format(header)
        else:
            text = header
        for i, joint_pck in enumerate(pck[1]):
            text += "{:.1f} & ".format(round(joint_pck[0] * 100, 1))
        text += "{:.1f} \\\\ ".format(round(pck[0][0] * 100, 1))
    if pck_full is not None:
        # text += "\n"
        if len(pck) != 1:
            text = text[:-3]
        if len(pck) == 1:
            text += " & {:.1f}".format(round(pck_full[0] * 100, 1))
        else:
            text += " & {:.1f} \\\\ ".format(round(pck_full[0][0] * 100, 1))
    return text