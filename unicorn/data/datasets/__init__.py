#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection
# SOT
from .coco_sot import COCOSOTDataset
from .lasot import Lasot
from .got10k import Got10k
from .tracking_net import TrackingNet
from .omni_data import OmniDataset
# MOT
from .mot import MOTDataset
from .mot_omni import MOTOmniDataset
from .bdd import BDDDataset
from .bdd_omni import BDDOmniDataset
# SOT-MOT
from .mosaicdetection_uni import MosaicDetectionUni
from .omni_data import OmniDatasetPlus
# Instance Segmentation
from .coco_inst import COCOInsDataset
from .mosaicdetection import MosaicDetectionIns
# MOTS
from .coco_mots import COCOMOTSDataset
from .mots_mot import MOTSMOTDataset
from .bdd_omni_mots import BDDOmniMOTSDataset
# VOS
from .saliency import SaliencyDataset
from .davis import DAVISDataset
from .youtube_vos import YoutubeVOSDataset
# SOT-MOT-VOS-MOTS
from .mosaicdetection_uni import MosaicDetectionUni4tasks