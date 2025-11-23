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
