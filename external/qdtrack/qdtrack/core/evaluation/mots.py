import time
from collections import defaultdict

import numpy as np
import pandas as pd

import motmetrics as mm
import pycocotools.mask as mask_util

from .mot_pcan import (xyxy2xywh, preprocessResult, aggregate_eval_results)

# note that there is no +1

def mask_iou_matrix(objs, hyps, max_iou=1.):
    if len(objs) == 0 or len(hyps) == 0:
        return np.empty((0, 0))
    iscrowd = np.zeros(len(objs))
    C = 1 - mask_util.iou(hyps, objs, iscrowd)
    C[C > max_iou] = np.nan
    return C.transpose()


def eval_mots(api, anns, all_results, split_camera=False, class_average=False):
    print('Evaluating BDD Results...')
    assert len(all_results) == len(anns['images'])
    t = time.time()

    cats_mapping = {k['id']: k['id'] for k in anns['categories']}

    preprocessResult(all_results, anns, cats_mapping)
    anns['annotations'] = [
        a for a in anns['annotations']
        if not (a['iscrowd'] or a.get('ignore', False))
    ]

    # fast indexing
    annsByAttr = defaultdict(lambda: defaultdict(list))

    for i, bbox in enumerate(anns['annotations']):
        annsByAttr[bbox['image_id']][cats_mapping[bbox['category_id']]].append(
            i)

    box_track_acc = defaultdict(lambda: defaultdict())
    seg_track_acc = defaultdict(lambda: defaultdict())
    global_instance_id = 0
    num_instances = 0
    cat_ids = np.unique(list(cats_mapping.values()))
    video_camera_mapping = dict()
    for cat_id in cat_ids:
        for video in anns['videos']:
            box_track_acc[cat_id][video['id']] = mm.MOTAccumulator(auto_id=True)
            seg_track_acc[cat_id][video['id']] = mm.MOTAccumulator(auto_id=True)
            if split_camera:
                video_camera_mapping[video['id']] = video['camera_id']

    for img, results in zip(anns['images'], all_results):
        img_id = img['id']

        if img['frame_id'] == 0:
            global_instance_id += num_instances
        if len(list(results.keys())) > 0:
            num_instances = max([int(k) for k in results.keys()]) + 1

        pred_bboxes, pred_ids = defaultdict(list), defaultdict(list)
        pred_segms = defaultdict(list)
        for instance_id, result in results.items():
            _bbox = xyxy2xywh(result['bbox'])
            _cat = cats_mapping[result['label'] + 1]
            pred_bboxes[_cat].append(_bbox)
            instance_id = int(instance_id) + global_instance_id
            pred_ids[_cat].append(instance_id)
            pred_segms[_cat].append(result['segm'])

        gt_bboxes, gt_ids = defaultdict(list), defaultdict(list)
        gt_segms = defaultdict(list)
        for cat_id in cat_ids:
            for i in annsByAttr[img_id][cat_id]:
                ann = anns['annotations'][i]
                gt_bboxes[cat_id].append(ann['bbox'])
                gt_ids[cat_id].append(ann['instance_id'])
                gt_segms[cat_id].append(api.annToRLE(ann))
            box_distances = mm.distances.iou_matrix(
                gt_bboxes[cat_id], pred_bboxes[cat_id], max_iou=0.5)
            box_track_acc[cat_id][img['video_id']].update(gt_ids[cat_id],
                                                          pred_ids[cat_id],
                                                          box_distances)
            seg_distances = mask_iou_matrix(
                gt_segms[cat_id], pred_segms[cat_id], max_iou=0.5)
            seg_track_acc[cat_id][img['video_id']].update(gt_ids[cat_id],
                                                          pred_ids[cat_id],
                                                          seg_distances)

    def _eval_summary(track_acc, eval_name):
        empty_cat = []
        for cat, video_track_acc in track_acc.items():
            for vid, v in video_track_acc.items():
                if len(v._events) == 0:
                    empty_cat.append([cat, vid])
        for cat, vid in empty_cat:
            track_acc[cat].pop(vid)

        names, acc = [], []
        for cat, video_track_acc in track_acc.items():
            for vid, v in video_track_acc.items():
                name = '{}_{}'.format(cat, vid)
                if split_camera:
                    name += '_{}'.format(video_camera_mapping[vid])
                names.append(name)
                acc.append(v)

        metrics = [
            'mota', 'motp', 'num_misses', 'num_false_positives', 'num_switches',
            'mostly_tracked', 'mostly_lost', 'idf1'
        ]

        print(f'Evaluating {eval_name} tracking...')
        mh = mm.metrics.create()
        summary = mh.compute_many(
            acc,
            metrics=[
                'num_objects', 'motp', 'num_detections', 'num_misses',
                'num_false_positives', 'num_switches', 'mostly_tracked',
                'mostly_lost', 'idtp', 'num_predictions'
            ],
            names=names,
            generate_overall=False)
        if split_camera:
            summary['camera_id'] = summary.index.str.split('_').str[-1]
            for camera_id, summary_ in summary.groupby('camera_id'):
                print('\nEvaluating camera ID: ', camera_id)
                aggregate_eval_results(
                    summary_,
                    metrics,
                    list(track_acc.keys()),
                    mh,
                    generate_overall=True,
                    class_average=class_average)

        print('\nEvaluating overall results...')
        summary = aggregate_eval_results(
            summary,
            metrics,
            list(track_acc.keys()),
            mh,
            generate_overall=True,
            class_average=class_average)
        out = {k: v for k, v in summary.to_dict().items()}

    # eval for track
    print('Generating matchings and summary...')
    box_track_out = _eval_summary(box_track_acc, 'box')
    seg_track_out = _eval_summary(seg_track_acc, 'seg')
    out = dict(box_track=box_track_out, seg_track=seg_track_out)
    print('Evaluation finsihes with {:.2f} s'.format(time.time() - t))

    return out
