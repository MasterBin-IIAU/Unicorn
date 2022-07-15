_base_ = './qdtrack-frcnn_r50_fpn_12e_bdd100k.py'
# model settings
model = dict(
    type='QuasiDenseMaskRCNN',
    pretrained='torchvision://resnet50',
    roi_head=dict(
        type='QuasiDenseSegRoIHead',
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHeadPlus',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    tracker=dict(type='QuasiDenseSegEmbedTracker'),
    # model training and testing settings
    train_cfg = dict(
        rcnn=dict(mask_size=28)),
    test_cfg = dict(
        rcnn=dict(mask_thr_binary=0.5))
)

# dataset settings
dataset_type = 'BDDVideoDataset'
data_root = 'datasets/bdd/'
img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_ins_id=True,
         with_mask=True),
    dict(type='SeqResize', img_scale=(1296, 720), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices', 'gt_masks'],
        ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1296, 720),
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
    samples_per_gpu=2, # ori is 2
    workers_per_gpu=2, # ori is 2
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'labels/seg_track_20/seg_track_train_cocoformat.json',
            img_prefix=data_root + 'images/seg_track_20/train',
            key_img_sampler=dict(interval=1),
            ref_img_sampler=dict(num_ref_imgs=1, scope=3, method='uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + 'labels/ins_seg/polygons/ins_seg_train_cocoformat.json',
            img_prefix=data_root + 'images/10k/train',
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'labels/seg_track_20/seg_track_val_cocoformat.json',
        img_prefix=data_root + 'images/seg_track_20/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'labels/seg_track_20/seg_track_val_cocoformat.json',
        #ann_file=data_root + 'labels/seg_track_20/seg_track_test_cocofmt.json',
        img_prefix=data_root + 'images/seg_track_20/val',
        #img_prefix=data_root + 'images/seg_track_20/test',
        pipeline=test_pipeline))
        
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # ori lr=0.01
load_from = './ckpts/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth'
resume_from = None
evaluation = dict(metric=['bbox', 'segm', 'segtrack'], interval=12)
