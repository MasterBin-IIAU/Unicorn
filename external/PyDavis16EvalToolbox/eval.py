# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import warnings

import cv2
import numpy as np
import yaml
from joblib import delayed, Parallel
from prettytable import PrettyTable as ptable

from metrics import f_boundary, jaccard

my_parser = argparse.ArgumentParser(
    description="The code is based on `https://github.com/davisvideochallenge/davis`",
    epilog="Enjoy the program! :)",
    allow_abbrev=False,
)
my_parser.version = "1.0.0"
my_parser.add_argument("-v", "--version", action="version")
my_parser.add_argument(
    "--name_list_path",
    default="/home/lart/Datasets/VideoSeg/DAVIS-2017-trainval-480p/DAVIS/ImageSets/2016/val.txt",
    type=str,
    help="the information file of DAVIS 2016 Dataset",
)
my_parser.add_argument(
    "--mask_root",
    default="/home/lart/Datasets/VideoSeg/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p",
    type=str,
    help="the annotation folder of DAVIS 2016 Dataset",
)
my_parser.add_argument(
    "--pred_path",
    default="/home/lart/coding/USVideoSeg/output/HDFNet_WSGNR50_V1/pre",
    type=str,
    help="the prediction folder of the method",
)
my_parser.add_argument(
    "--save_path",
    default="./output/HDFNet_WSGNR50_V1.pkl",
    type=str,
    help="the file path for saving evaluation results",
)
my_parser.add_argument(
    "--ignore_head",
    default="True",
    choices=["True", "False"],
    type=str,
    help="whether to ignore the first frame during evaluation",
)
my_parser.add_argument(
    "--ignore_tail",
    default="True",
    choices=["True", "False"],
    type=str,
    help="whether to ignore the last frame during evaluation",
)
my_parser.add_argument(
    "--n_jobs",
    default=2,
    type=int,
    help="the number of jobs for parallel evaluating the performance",
)


def print_all_keys(data_dict, level: int = 0):
    level += 1
    if isinstance(data_dict, dict):
        for k, v in data_dict.items():
            print(f" {'|=' * level}>> {k}")
            print_all_keys(v, level=level)
    elif isinstance(data_dict, (list, tuple)):
        for item in data_dict:
            print_all_keys(item, level=level)
    else:
        return


def get_eval_video_name_list_from_yml(path: str, data_set: str) -> list:
    with open(path, encoding="utf-8", mode="r") as f_stream:
        data_info_dict = yaml.load(f_stream, Loader=yaml.FullLoader)

    eval_video_name_list = []
    for video_dict in data_info_dict["sequences"]:
        if video_dict["set"] == data_set:
            eval_video_name_list.append(video_dict["name"])
    return eval_video_name_list


def get_mean_recall_decay_for_video(per_frame_values):
    """Compute mean,recall and decay from per-frame evaluation.

    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values[1:-1] > 0.5)

    # Compute decay as implemented in Matlab
    per_frame_values = per_frame_values[1:-1]  # Remove first frame

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i] : ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


def _read_and_eval_file(mask_video_path: str, pred_video_path: str, frame_name: str):
    frame_mask_path = os.path.join(mask_video_path, frame_name)
    frame_pred_path = os.path.join(pred_video_path, frame_name)
    frame_mask = cv2.imread(frame_mask_path, 0)  # h, w
    frame_pred = cv2.imread(frame_pred_path, 0)
    binary_frame_mask = (frame_mask > 0).astype(np.float32)
    binary_frame_pred = (frame_pred > 0).astype(np.float32)
    J_score = jaccard.db_eval_iou(
        annotation=binary_frame_mask, segmentation=binary_frame_pred
    )
    F_score = f_boundary.db_eval_boundary(
        foreground_mask=binary_frame_pred, gt_mask=binary_frame_mask
    )
    return J_score, F_score


def _eval_video_sequence(
    method_pre_path: str,
    mask_data_root: str,
    video_name: str,
    ignore_head: bool,
    ignore_tail: bool,
):
    print(f"processing {video_name}...")

    mask_video_path = os.path.join(mask_data_root, video_name)
    pred_video_path = os.path.join(method_pre_path, video_name)

    mask_frame_path_list = sorted(os.listdir(mask_video_path))
    if ignore_head:
        mask_frame_path_list = mask_frame_path_list[1:]
    if ignore_tail:
        mask_frame_path_list = mask_frame_path_list[:-1]

    frame_score_list = [
        _read_and_eval_file(
            mask_video_path=mask_video_path,
            pred_video_path=pred_video_path,
            frame_name=frame_name,
        )
        for frame_name in mask_frame_path_list
    ]
    if ignore_head:
        frame_score_list = [[np.nan, np.nan]] + frame_score_list
    if ignore_tail:
        frame_score_list += [[np.nan, np.nan]]
    frame_score_array = np.asarray(frame_score_list)
    M, O, D = zip(
        *[
            get_mean_recall_decay_for_video(frame_score_array[:, i])
            for i in range(frame_score_array.shape[1])
        ]
    )
    return {
        video_name: {
            "pre_frame": frame_score_array,
            "mean": np.asarray(M),
            "recall": np.asarray(O),
            "decay": np.asarray(D),
        }
    }


def get_method_score_dict(
    method_pre_path: str,
    mask_data_root: str,
    video_name_list: list,
    ignore_head: bool = True,
    ignore_tail: bool = True,
    n_jobs: int = 2,
):
    video_score_list = Parallel(n_jobs=n_jobs)(
        delayed(_eval_video_sequence)(
            method_pre_path=method_pre_path,
            mask_data_root=mask_data_root,
            video_name=video_name,
            ignore_head=ignore_head,
            ignore_tail=ignore_tail,
        )
        for video_name in video_name_list
    )
    video_score_dict = {
        list(kv.keys())[0]: list(kv.values())[0] for kv in video_score_list
    }
    return video_score_dict


def get_method_average_score_dict(method_score_dict: dict):
    # average_score_dict = {"total": 0, "mean": 0, "recall": 0, "decay": 0}
    average_score_dict = {"Average": {"mean": 0, "recall": 0, "decay": 0}}
    for k, v in method_score_dict.items():
        # average_score_item = np.nanmean(v["pre_frame"], axis=0)
        # average_score_dict[k] = average_score_item
        average_score_dict[k] = {
            "mean": v["mean"],
            "recall": v["recall"],
            "decay": v["decay"],
        }
        # average_score_dict["total"] += average_score_item
        average_score_dict["Average"]["mean"] += v["mean"]
        average_score_dict["Average"]["recall"] += v["recall"]
        average_score_dict["Average"]["decay"] += v["decay"]
    # average_score_dict['Average']["total"] /= len(method_score_dict)
    average_score_dict["Average"]["mean"] /= len(method_score_dict)
    average_score_dict["Average"]["recall"] /= len(method_score_dict)
    average_score_dict["Average"]["decay"] /= len(method_score_dict)
    return average_score_dict


def save_to_file(data, save_path: str):
    with open(save_path, mode="wb") as f:
        pickle.dump(data, f)


def read_from_file(file_path: str):
    with open(file_path, mode="rb") as f:
        data = pickle.load(f)
    return data


def convert_data_dict_to_table(data_dict: dict, video_name_list: list):
    table = ptable(["Video", "J(M)", "J(O)", "J(D)", "F(M)", "F(O)", "F(D)"])
    for video_name in video_name_list:
        table.add_row(
            [video_name]
            + [
                f"{data_dict[video_name][x][y]: .3f}"
                for y in range(2)
                for x in ["mean", "recall", "decay"]
            ]
        )
    return "\n" + str(table) + "\n"


def get_eval_video_name_list_from_txt(path: str) -> list:
    name_list = []
    with open(path, encoding="utf-8", mode="r") as f:
        for line in f:
            line = line.strip()
            if line:
                name_list.append(line)
    return name_list


def eval_method_from_data(
    method_pre_path: str,
    mask_data_root: str,
    ignore_head: bool,
    ignore_tail: bool,
    name_list_path: str,
    save_path: str = "./output/average.pkl",
    n_jobs: int = 2,
):
    """
    根据给定方法的预测结果来评估在davis 2016上的性能
    :param method_pre_path: 模型预测结果，该路径下包含各个视频预测的结果，与Annotations文件夹布局一致
    :param mask_data_root: davis 2016的Annotations文件夹
    :param ignore_head: 评估时是否忽略第一帧
    :param ignore_tail: 评估时是否忽略最后一帧
    :param name_list_path: davis 2016数据集的信息文件（db_info.yml）或者是2017中提供的 2016/val.txt
    :param save_path: 保存导出的模型评估结果的文件路径
    :param n_jobs: 多进程评估时使用的进程数
    """
    if name_list_path.endswith(".yml") or name_list_path.endswith(".yaml"):
        # read yaml and get the list  that will be used to eval the model
        eval_video_name_list = get_eval_video_name_list_from_yml(
            path=name_list_path, data_set="test"
        )
    elif name_list_path.endswith(".txt"):
        eval_video_name_list = get_eval_video_name_list_from_txt(path=name_list_path)
    else:
        raise ValueError

    # tervese the each video
    method_score_dict = get_method_score_dict(
        method_pre_path=method_pre_path,
        mask_data_root=mask_data_root,
        video_name_list=eval_video_name_list,
        ignore_head=ignore_head,
        ignore_tail=ignore_tail,
        n_jobs=n_jobs,
    )
    # get the average score
    average_score_dict = get_method_average_score_dict(
        method_score_dict=method_score_dict
    )

    if save_path != None:
        save_to_file(data=average_score_dict, save_path=save_path)

    # show the results
    eval_video_name_list += ["Average"]
    table_str = convert_data_dict_to_table(
        data_dict=average_score_dict, video_name_list=eval_video_name_list
    )
    print(table_str)


def show_results_from_data_file(file_path: str = "./output/average.pkl"):
    """
    展示给定的模型评估结果文件中包含的模型的结果
    :param file_path: 保存导出的模型评估结果的文件路径
    """
    average_score_dict = read_from_file(file_path=file_path)

    eval_video_name_list = list(average_score_dict.keys())
    eval_video_name_list[0], eval_video_name_list[-1] = (
        eval_video_name_list[-1],
        eval_video_name_list[0],
    )
    # show the results
    table_str = convert_data_dict_to_table(
        data_dict=average_score_dict, video_name_list=eval_video_name_list
    )
    print(table_str)


if __name__ == "__main__":
    args = my_parser.parse_args()
    eval_method_from_data(
        method_pre_path=args.pred_path,
        mask_data_root=args.mask_root,
        ignore_tail=True if args.ignore_tail == "True" else False,
        ignore_head=True if args.ignore_head == "True" else False,
        name_list_path=args.name_list_path,
        save_path=args.save_path,
        n_jobs=args.n_jobs,
    )
    # show_results_from_data_file("./output/dybinary_ave.pkl")
    # show_results_from_data_file("./output/HDFNet_WSGNR50_V1.pkl")
    # show_results_from_data_file("./output/matnet_ave.pkl")
