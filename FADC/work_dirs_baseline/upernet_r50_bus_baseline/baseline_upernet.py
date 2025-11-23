E:\S3\OT\FADC\FADC\configs\_base_\models\upernet_r50_bus.py
# ========================================
# === Model: UPerNet with ResNet50 Backbone (FADC Faithful) ===
# ========================================

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),  # FADC-style dilation
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=3,  # (background, benign, malignant)
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    loss_weight=1.0,
    class_weight=[0.2, 1.0, 2.5] )

    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[0.2, 1.0, 2.5]
        )

    ),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ========================================
# === Basic Runtime Settings (Non-duplicated) ===
# ========================================

work_dir = './work_dirs/upernet_r50_bus'

E:\S3\OT\FADC\FADC\configs\_base_\datasets\bus_uclm.py
# BUS-UCLM Breast Ultrasound Dataset (2-class version)
# Compatible with MMSegmentation 0.25.0

dataset_type = 'CustomDataset'
data_root = r"E:\S3\OT\FADC\FADC\data\BUS_RCNN2"

# === 2-CLASS SETUP ===
classes = ('background', 'benign', 'malignant')
palette = [
    [0, 0, 0],      # background
    [0, 255, 0],    # benign
    [255, 0, 0]     # malignant
]

reduce_zero_label = False

# === Pipelines ===
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# === Dataset definitions ===
# === Dataset definitions ===
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        img_suffix='',
        seg_map_suffix='',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        split='split/train.txt'     # ✅ added
    ),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        img_suffix='',
        seg_map_suffix='',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        split='split/val.txt'       # ✅ added
    ),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='masks',
        img_suffix='',
        seg_map_suffix='',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=False,
        split='split/val.txt'       # ✅ you can also make a separate test split later
    )
)



E:\S3\OT\FADC\FADC\configs\_base_\schedules\schedule_80k.py
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False
)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')

E:\S3\OT\FADC\FADC\configs\_base_\default_runtime.py
# yapf:disable
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='CustomizedTextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

E:\S3\OT\FADC\FADC\configs\baseline_upernet.py
# ==========================================
# Baseline UPerNet-R50 for BUS-UCLM
# ==========================================

_base_ = [
    './_base_/models/upernet_r50_bus.py',
    './_base_/datasets/bus_uclm.py',
    './_base_/schedules/schedule_80k.py',
    './_base_/default_runtime.py'
]

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=1))


# Override ONLY what you must change for BUS-UCLM
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',

    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),

    decode_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 1.0, 2.5],
            loss_weight=1.0
        )
    ),

    auxiliary_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 1.0, 2.5],
            loss_weight=0.4
        )
    )
)

# Where checkpoint will be saved
work_dir = './work_dirs/upernet_r50_bus_baseline'
