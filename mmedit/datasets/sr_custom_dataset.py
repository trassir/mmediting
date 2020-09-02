import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS

import numpy as np


@DATASETS.register_module()
class SRCustomDataset(BaseSRDataset):
    """General paired image folder dataset for image restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "folder mode", which needs to specify the lq folder path and gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically. You can also specify the
    filename template to match the lq and gt pairs.

    For example, we have two folders with the following structures:

    ::

        data_root
        ├── lq
        │   ├── 0001_x4.png
        │   ├── 0002_x4.png
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png

    then, you need to set:

    .. code-block:: python

        lq_folder = data_root/lq
        gt_folder = data_root/gt
        filename_tmpl = '{}_x4'

    Args:
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Default: '{}'.
    """

    def __init__(self,
                 gt_folder,
                 pipeline,
                 scale,
                 test_mode=False,
                 filename_tmpl='{}'):
        super(SRCustomDataset, self).__init__(pipeline, scale, test_mode)
        self.gt_folder = str(gt_folder)
        self.filename_tmpl = filename_tmpl
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for SR dataset.

        It loads the LQ and GT image path from folders.

        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        gt_paths = self.scan_folder(self.gt_folder)
        np.random.shuffle(gt_paths)
        for gt_path in gt_paths:
            data_infos.append(dict(gt_path=gt_path, lq_path=gt_path))
        return data_infos
