import random
# import torch.utils.data
from .datasets_wrapper import Dataset
import numpy as np

class OmniDataset(Dataset):
    """ general dataset class which is compatible with all kinds of datasets
    """

    def __init__(self, img_size, datasets, p_datasets, samples_per_epoch):
        super().__init__(img_size)
        self.datasets = datasets
        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]
        self.samples_per_epoch = samples_per_epoch
        self.datasets_id_list = list(range(len(self.datasets)))

    def __len__(self):
        return self.samples_per_epoch
    
    def pull_item(self, idx, num_frames=2):
        if len(self.datasets_id_list) != 1:
            dataset_id = random.choices(self.datasets_id_list, self.p_datasets)[0]
        else:
            dataset_id = 0
        dataset = self.datasets[dataset_id]
        data, seq_id, obj_id = dataset.pull_item(idx, num_frames)
        return data, dataset_id, seq_id, obj_id
    
    def pull_item_id(self, dataset_id, seq_id, obj_id, num_frames):
        # get data according to dataset_id and seq_id
        dataset = self.datasets[dataset_id]
        return dataset.pull_item_id(seq_id, obj_id, num_frames)

class OmniDatasetPlus(Dataset):
    """ general dataset class which is compatible with all kinds of datasets
    """

    def __init__(self, img_size, datasets, p_datasets, samples_per_epoch, mode="joint", fix=False, fix_id=None):
        super().__init__(img_size)
        self.datasets = datasets
        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]
        self.samples_per_epoch = samples_per_epoch
        self.task_id_list = list(range(len(self.datasets)))
        assert mode in ["joint", "alter"]
        self.mode = mode
        if self.mode == "alter":
            self.cur_task_id = 0
            self.altered = False
        self.fix = fix
        self.fix_id = fix_id
        if self.fix and (self.fix_id is not None):
            self.cur_task_id = fix_id
    
    def alter_task(self):
        if not self.fix:
            self.cur_task_id = (self.cur_task_id + 1) % len(self.task_id_list)
            self.altered = True
            # print("altered %d, cur_task_id %d" %(self.altered, self.cur_task_id))

    def __len__(self):
        return self.samples_per_epoch
    
    def pull_item(self, idx, num_frames=2):
        if self.mode == "joint":
            if len(self.task_id_list) != 1:
                task_index = random.choices(self.task_id_list, self.p_datasets)[0]
            else:
                task_index = 0
        else:
            task_index = self.cur_task_id
            # print("altered %d, task_index %d" %(self.altered, task_index))
        dataset = self.datasets[task_index]
        data, _, seq_id, obj_id = dataset.pull_item(idx, num_frames)
        task_id = task_index + 1 # SOT=1, MOT=2, VOS=3, MOTS=4
        return data, np.array([task_id]), seq_id, obj_id
    
    def pull_item_id(self, dataset_id, seq_id, obj_id, num_frames):
        # get data according to dataset_id and seq_id
        dataset = self.datasets[dataset_id]
        return dataset.pull_item_id(seq_id, obj_id, num_frames)