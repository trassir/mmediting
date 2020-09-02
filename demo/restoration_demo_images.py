import argparse
from itertools import chain
from pathlib import Path

import mmcv
import torch
import numpy as np
from detector_utils import Size2D

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_path', type=Path, help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def restoration_inference(model, img):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    data = dict(gt_path=img, lq_path=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    return result


def main():
    import cv2
    from scipy.spatial.distance import cosine
    from demo.face_evolve_model import FaceEvolveModel

    args = parse_args()

    model = init_model(args.config, args.checkpoint,
                       device=torch.device('cuda', args.device))
    reid_model = FaceEvolveModel(backbone_type='IR_50',
                                 checkpoint_path=Path('./backbone_ir50_asia.pth'),
                                 input_size=Size2D(112, 112),
                                 feat_num=512)

    assert args.img_path.exists()
    filenames = chain.from_iterable([args.img_path.rglob(f'*.{ext}')
                                     for ext in {'jpg', 'png'}])

    cv2.namedWindow('lq', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('gt', cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow('pred', cv2.WINDOW_KEEPRATIO)
    for filename in filenames:
        results = restoration_inference(model, str(filename))
        pred = tensor2img(results['output']).astype(np.uint8)
        lq = tensor2img(results['lq']).astype(np.uint8)
        gt = tensor2img(results['gt']).astype(np.uint8)

        lq_v, gt_v, pred_v = reid_model.get_feature([lq[..., ::-1],
                                                     gt[..., ::-1],
                                                     pred[..., ::-1]])
        print(f'\nlq with gt: {cosine(lq_v, gt_v)}')
        print(f'pred with gt: {cosine(pred_v, gt_v)}')

        mmcv.imshow(lq, 'lq', wait_time=1)
        mmcv.imshow(gt, 'gt', wait_time=1)
        mmcv.imshow(pred, 'pred')


if __name__ == '__main__':
    main()
