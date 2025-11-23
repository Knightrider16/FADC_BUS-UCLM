# ========================
# FORCE FADC REGISTRATION
# ========================
# Do NOT import HorNet. Only import conv_custom to register AdaDilatedConv.
import FADC.models.conv_custom

custom_imports = dict(
    imports=['FADC.models.conv_custom'],   # ensures AdaptiveDilatedConv registers
    allow_failed_imports=False
)

# IMPORTANT: Enable plugins so MMDetection loads this folder properly.
plugin = True
plugin_dir = 'FADC'


# ========================
# BASE CONFIGS
# ========================
_base_ = [
    './base/mask_rcnn_r50_fpn_1x_coco.py',   # ← from FADC/base
    '../../_base_/datasets/bus_uclm.py',     # ← your dataset config
    '../../_base_/schedules/schedule_80k.py',# ← choose your schedule
    '../../_base_/default_runtime.py'
]


# ========================
# DATASET SETTINGS
# ========================
dataset_type = 'COCODataset'
data_root = r"E:\S3\OT\FADC\FADC\data\BUS_COCO"

classes = ('benign', 'malignant')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,   # Windows-safe

    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_train.json',
        img_prefix=data_root + '/images/',
        classes=classes
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_val.json',
        img_prefix=data_root + '/images/',
        classes=classes
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations_val.json',
        img_prefix=data_root + '/images/',
        classes=classes
    ),
)


# ========================
# MODEL (FADC in BACKBONE)
# ========================
model = dict(
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        dcn=dict(
            type='AdaDilatedConv',
            offset_freq=None,
            epsilon=1e-4,
            use_zero_dilation=False,
            deformable_groups=1,
            padding_mode='repeat',
            kernel_decompose='both',
            pre_fs=True,
            conv_type='conv',
            fs_cfg=dict(
                k_list=[2, 4, 8],
                fs_feat='feat',
                lowfreq_att=False,
                lp_type='freq',
                act='sigmoid',
                spatial='conv',
                spatial_group=1,
            ),
            sp_att=False,
            fallback_on_stride=False
        ),
        stage_with_dcn=(False, True, True, True),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False
    ),

    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=2,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[1.0, 3.0]
            )
        ),
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=2,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True,
                class_weight=[1.0, 3.0]
            )
        )
    )
)


# ========================
# OPTIMIZER (FADC Recommended)
# ========================
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructorHorNet',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        decay_rate=0.9,
        decay_type='stage_wise',
        num_layers=12
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))


# ========================
# LR POLICY
# ========================
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)


# ========================
# TRAINING CONTROL
# ========================
runner = dict(type='IterBasedRunner', max_iters=80000)

checkpoint_config = dict(interval=4000, max_keep_ckpts=2)
evaluation = dict(interval=4000, metric=['bbox', 'segm'])
