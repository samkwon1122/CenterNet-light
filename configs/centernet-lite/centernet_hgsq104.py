_base_ = [
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../centernet/centernet_tta.py'
]

dataset_type = 'VOCDataset'         # dataset 타입과 경로 지정
data_root = '/mmdetection/data/VOCdevkit/'
backend_args = None

# model settings
model = dict(
    type='CenterNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HourglassSqNet',      # HourglassSqNet 사용
        downsample_times=4,         # Sq모델은 다운샘플링 4번
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 512],   # 다운샘플링 횟수 변경 되었으므로 수정
        stage_blocks=[2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,                      # Hourglass는 neck 사용 안 함
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=20,             # VOC dataset 이므로 클래스 20개
        in_channels=256,            # head에 들어가는 채널 수 명시
        feat_channels=256,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        # The cropped images are padded into squares during training,
        # but may be less than crop_size.
        crop_size=(512, 512),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    # Make sure the output is always crop_size.
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        to_float32=True),
    # don't need Resize
    dict(
        type='RandomCenterCropPad',
        ratios=None,
        border=None,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_mode=True,
        test_pad_mode=['logical_or', 31],
        test_pad_add_pix=1),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'border'))
]

# Use RepeatDataset to speed up training
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='ConcatDataset',
            # VOCDataset will add different `dataset_type` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args)
            ])))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_ = True,
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.00025))    # optimizer와 learning rate 지정

max_epochs = 10     # epoch 수 지정 (repeat 5번 이므로 10 epochs이면 실질적으로 50 epochs)
# learning policy
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],  # 8*5=40 epoch에서 learning rate 감소
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)  # the real epoch is 10*5=50

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (16 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
