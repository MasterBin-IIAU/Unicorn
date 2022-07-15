import argparse
import os

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets import build_dataset
from qdtrack.core.to_bdd100k import preds2bdd100k


def parse_args():
    parser = argparse.ArgumentParser(description='qdtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--res', help='output result file')
    parser.add_argument(
        '--bdd-dir',
        type=str,
        help='path to the folder that will contain files in bdd100k format')
    parser.add_argument(
        '--coco-file',
        type=str,
        help='path to that json file that is in COCO submission format')
    parser.add_argument(
        '--task',
        type=str,
        nargs='+',
        help='task types',
        choices=['det', 'ins_seg', 'box_track', 'seg_track'])
    parser.add_argument(
        '--nproc',
        type=int,
        help='number of process for mask merging')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isfile(args.res):
        raise ValueError('The result file does not exist.')

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('USE_MMDET', False):
        from mmdet.datasets import build_dataloader
    else:
        from qdtrack.datasets import build_dataloader

    # build the dataloader
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    print(f'\nLoading results from {args.res}')
    results = mmcv.load(args.res)

    if args.coco_file:
        dataset.format_results(results, jsonfile_prefix=args.coco_file)
    if args.bdd_dir:
        print("converting results to bdd100k...")
        preds2bdd100k(
            dataset, results, args.task, out_base=args.bdd_dir, nproc=args.nproc)
    print("Conversion is done.")

if __name__ == '__main__':
    main()
