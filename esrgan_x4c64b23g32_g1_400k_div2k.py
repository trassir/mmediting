exp_name = 'esrgan_x4c64b23g32_g1_400k_div2k'

scale = 4
# model settings
model = dict(
    type='ESRGAN',
    generator=dict(
        type='RRDBNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=23,
        growth_channels=32),
    discriminator=dict(type='ModifiedVGG', in_channels=3, mid_channels=64),
    pixel_loss=dict(type='L1Loss', loss_weight=1e-2, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'34': 1.0},
        vgg_type='vgg19',
        perceptual_weight=1.0,
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-3,
        real_label_val=1.0,
        fake_label_val=0),
    pretrained=None,
)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SRCustomDataset'
val_dataset_type = 'SRCustomDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='ImgaugAugmentation', width_gt=192, height_gt=192, width_lq=48, height_lq=48, debug=False),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='ImgaugAugmentation', width_gt=192, height_gt=192, width_lq=48, height_lq=48, debug=False),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    # train
    samples_per_gpu=24,
    workers_per_gpu=16,
    drop_last=True,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='/media/svakhreev/fast/sres_faces_data',
            pipeline=train_pipeline,
            scale=scale)),
    # val
    val_samples_per_gpu=1,
    val_workers_per_gpu=1,
    val=dict(
        type=val_dataset_type,
        gt_folder='/media/svakhreev/fast/sres_faces_data_test',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    # test
    test=dict(
        type=val_dataset_type,
        gt_folder='/media/svakhreev/fast/sres_faces_data_test',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),
    discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 400000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[50000, 100000, 200000, 300000],
    gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook', project='esrgan_face',
        #      config_filename='./esrgan_x4c64b23g32_g1_400k_div2k.py')
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = f'./work_dirs/{exp_name}/latest.pth'
resume_from = None
workflow = [('train', 1)]
