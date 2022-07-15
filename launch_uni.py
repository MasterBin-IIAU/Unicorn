#!/usr/bin/python3
import os
import argparse
import random
import time

"""Unified launch script for object detection, instance segmentation and four tracking tasks"""
def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--nproc_per_node', type=int, default=8, help="number of GPUs per node")  # num_gpu per node
    parser.add_argument('--batch', type=int, help="global batch size")
    parser.add_argument('--mode', type=str, choices=["multiple", "distribute"], default="multiple",
                        help="train on a single node or multiple nodes")
    parser.add_argument('--port', type=int, default=0)
    # The following args are required for "distributed" mode (training on multiple nodes)
    parser.add_argument('--master_address', type=str, help="address of the master node")
    parser.add_argument('--nnodes', type=int, help="the total number of nodes")
    parser.add_argument('--node_rank', type=int, help="the rank of the current node")

    args = parser.parse_args()

    return args


def _get_rand_port():
    hour = time.time() // 3600
    random.seed(int(hour))
    return random.randrange(40000, 60000)


def main():
    args = parse_args()
    # change to current dir
    prj_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(prj_dir)
    # Get port
    if args.port > 0:
        master_port = args.port
    else:  # This reduce the conflict possibility, but the port availablity is not guaranteed.
        master_port = _get_rand_port()
    # train
    file_name = "exps/default/%s" % args.name
    if args.mode == "multiple":
        train_cmd = "python3 tools/train.py -f %s -d %d -b %d -o --resume" % (file_name, args.nproc_per_node, args.batch)
    elif args.mode == "distribute":
        sub_cmd = "tools/train_dist.py -f %s -b %d -o --resume" % (file_name, args.batch)
        train_cmd = f'python3 -m torch.distributed.launch --nproc_per_node={args.nproc_per_node} \
        --nnodes={args.nnodes} --node_rank={args.node_rank} --master_addr={args.master_address} --master_port={master_port} \
        {sub_cmd}'
    else:
        raise ValueError("mode should be 'multiple' or 'distribute'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
