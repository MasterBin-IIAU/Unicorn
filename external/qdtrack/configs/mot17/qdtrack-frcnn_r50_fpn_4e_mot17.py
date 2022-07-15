# model settings
_base_ = '../_base_/qdtrack_faster_rcnn_r50_fpn.py'
# search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs']
model = dict(
    backbone=dict(norm_cfg=dict(requires_grad=False), style='caffe'),
    rpn_head=dict(bbox_coder=dict(clip_border=False)),
    roi_head=dict(
        bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
    tracker=dict(
        type='QuasiDenseEmbedTracker',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'),
    # model training and testing settings
    train_cfg=dict(embed=dict(assigner=dict(neg_iou_thr=0.5))))
# dataset settings
dataset_type = 'MOT17Dataset'
data_root = 'data/MOT17/'
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(
        type='SeqRandomCrop',
        share_params=False,
        crop_size=(1088, 1088),
        bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices'],
        ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1088, 1088),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=['pedestrian'],
        visibility_thr=-1,
        track_visibility_thr=-1,
        ann_file='data/MOT17/annotations/half-train_cocoformat.json',
        img_prefix='data/MOT17/train/',
        ref_img_sampler=dict(num_ref_imgs=1, scope=10, method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='data/MOT17/annotations/half-val_cocoformat.json',
        classes=['pedestrian'],
        img_prefix='data/MOT17/train/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='data/MOT17/annotations/half-val_cocoformat.json',
        img_prefix='data/MOT17/train/',
        classes=['pedestrian'],
        ref_img_sampler=None,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[3])
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 4
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric=['bbox', 'track'], interval=1)
load_from = 'ckpts/mmdet/faster_rcnn_r50_caffe_fpn_person_ap551.pth'
work_dir = 'work_dirs/MOT17/qdtrack-frcnn_r50_fpn_4e_mot17'
