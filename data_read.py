from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import os.path as osp
from collections import defaultdict


class ImageFolderTwo(Dataset):
    def __init__(self, db_path_1, db_path_2, transform_1=None, transform_2=None, instance_num=2):
        labels_1 = list(sorted(os.listdir(db_path_1)))
        labels_2 = list(sorted(os.listdir(db_path_2)))
        assert labels_1 == labels_2

        self.instance_num = instance_num
        self.id_to_items_1 = defaultdict(list)
        self.id_to_items_2 = defaultdict(list)
        count = 0
        for id, label in enumerate(labels_1):
            for name in  os.listdir(osp.join(db_path_1, label)):
                self.id_to_items_1[id].append([osp.join(db_path_1, label, name), id, 1])
                count += 1
            for name in  os.listdir(osp.join(db_path_2, label)):
                self.id_to_items_2[id].append([osp.join(db_path_2, label, name), id, 2])
                count += 1
        
        id_num_1 = len(self.id_to_items_1.keys())
        id_num_2 = len(self.id_to_items_2.keys())
        assert id_num_1 == id_num_2
        self.indices = list(range(0, id_num_1))
        self.repeat_num = count // (id_num_1 * instance_num)

        self.transform_1 = transform_1
        self.transform_2 = transform_2

        self.shuffle_items()

    def shuffle_items(self):
        np.random.shuffle(self.indices)
        self.item_list = []
        for _ in range(self.repeat_num):
            for id in self.indices:
                self.item_list.append(self.id_to_items_1[id][0])
                id_to_item_2 = self.id_to_items_2[id]
                idxes = list(range(0, len(id_to_item_2)))
                np.random.shuffle(idxes)
                for num, idx in enumerate(idxes):
                    if num + 1 == self.instance_num:
                        break
                    self.item_list.append(id_to_item_2[idx])

    def __getitem__(self, idx):
        path, id, view = self.item_list[idx]

        img = Image.open(open(path, 'rb')).convert('RGB')
        id = int(id)

        if view == 1 and self.transform_1 is not None:
            img = self.transform_1(img)
        if view == 2 and self.transform_2 is not None:
            img = self.transform_2(img)
        
        return img, id

    def __len__(self):
        return len(self.item_list)


if __name__ == "__main__":
    data_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/benchmarks/reid/University-Release/train"
    dataset = ImageFolderTwo(db_path_1=os.path.join(data_dir, 'satellite'), db_path_2=os.path.join(data_dir, 'drone'), instance_num=3)
    dataset.shuffle_items()
    dataset.shuffle_items()
    dataset.shuffle_items()
    dataset.shuffle_items()

