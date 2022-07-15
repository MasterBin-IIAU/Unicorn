import argparse
import os

import mmcv
import torch
from mmcv import Config
from mmdet.datasets import build_dataset
from qdtrack.datasets import BDDVideoDataset
from mmcv.runner import get_dist_info, init_dist
import cv2
from qdtrack.datasets import build_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='qdtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.test.test_mode = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # build the dataloader
    save_dir = "debug"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    rank, _ = get_dist_info()
    for idx, data in enumerate(data_loader):
        if idx == 1:
            break
        print(rank, data)
        # img = data["img"][0].numpy().transpose(1, 2, 0)
        # cv2.imwrite(os.path.join(save_dir, "%02d.jpg" % idx), img)


if __name__ == '__main__':
    main()
