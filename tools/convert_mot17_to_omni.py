from unicorn.data import get_unicorn_datadir
from unicorn.data.datasets import MOTDataset
import os
import json
import argparse

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--dataset_name', type=str, default="mot")
    parser.add_argument('--ori_json_file', type=str, default="train.json")
    parser.add_argument('--new_json_file', type=str, default="train_omni.json")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name
    ori_json_file = args.ori_json_file
    new_json_file = args.new_json_file
    data_dir = os.path.join(get_unicorn_datadir(), dataset_name)
    save_path = os.path.join(data_dir, "annotations", new_json_file)
    dataset = MOTDataset(data_dir=data_dir, json_file=ori_json_file)
    omni_json = {}
    for (res, img_info, file_name) in dataset.annotations:
        (height, width, frame_id, video_id, file_name) = img_info
        if video_id not in omni_json:
            omni_json[video_id] = {}
        omni_json[video_id][frame_id] = {"res": res.tolist(), "img_info": img_info, "file_name": file_name}
    json.dump(omni_json, open(save_path, "w"))
