# Copyright (c) OpenMMLab. All rights reserved.
# Frequency Adaptive Dilated Convolution (FADC) based UPerNet-R50
# Adapted for BUS-UCLM Breast Ultrasound Segmentation Dataset

_base_ = [
    '../_base_/models/upernet_r50_bus.py',
    '../_base_/datasets/bus_uclm.py',
    '../_base_/schedules/schedule_80k.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        with_cp=False,
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
        ),
        stage_with_dcn=(False, True, True, True),
    ),

    # ================= LOSS UPDATE =================
    decode_head=dict(
        num_classes=3,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_weight=1.0,
                class_weight=[0.05, 1.0, 6.0]   # Background, Benign, Malignant
            ),
            dict(
                type='DiceLoss',                # Helps malignant shape consistency
                loss_weight=1.5,
                smooth=1e-5,
                ignore_index=255
            )
        ]
    ),

    auxiliary_head=dict(
        num_classes=3,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_weight=0.4,
                class_weight=[0.05, 1.0, 6.0]
            )
        ]
    ),

    # ================= TEST MODE =================
    test_cfg=dict(
        mode='slide',
        crop_size=(512, 512),
        stride=(341, 341)
    )
)

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
    )
)

checkpoint_config = dict(max_keep_ckpts=2)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')

optimizer_config = dict(
    grad_clip=dict(max_norm=1, norm_type=2)
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1e-6,
    power=0.95,
    min_lr=0.0,
    by_epoch=False
)

runner = dict(type='IterBasedRunner', max_iters=80000)

fp16 = dict(loss_scale='dynamic')
