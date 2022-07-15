#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------
# Unicorn
# Copyright (c) 2022 ByteDance. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX)
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# ----------------------------------------------------------------------


import torch

class BasePrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = BasePrefetcher._record_stream_for_image

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


class DataPrefetcher(BasePrefetcher):

    def __init__(self, loader):
        super().__init__(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


"""Prefetcher for instance segmentation"""
class DataPrefetcherIns(BasePrefetcher):

    def __init__(self, loader):
        super().__init__(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _, self.next_mask = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_mask = None
            return
        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_mask = self.next_mask.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        mask = self.next_mask
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if mask is not None:
            mask.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, mask


class DataPrefetcherUniBox(BasePrefetcher):

    def __init__(self, loader):
        super().__init__(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_task_id = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_task_id = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_task_id = self.next_task_id.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        task_id = self.next_task_id
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if task_id is not None:
            task_id.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, task_id


class DataPrefetcherUniMask(BasePrefetcher):

    def __init__(self, loader):
        super().__init__(loader)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_task_id, self.next_mask = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_task_id = None
            self.next_mask = None
            return
        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_task_id = self.next_task_id.cuda(non_blocking=True)
            self.next_mask = self.next_mask.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        task_id = self.next_task_id
        mask = self.next_mask
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if task_id is not None:
            task_id.record_stream(torch.cuda.current_stream())
        if mask is not None:
            mask.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, task_id, mask
