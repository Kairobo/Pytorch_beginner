import os
import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms
from utils.visual_utils import *
from utils.data_utils import *


class CustomDataset(Dataset):

    def __init__(self, data_config = None):
        # ----- process data config
        if 'split' in data_config:
            self.split = data_config['split']
        else:
            self.split = 'test'

        if 'root_dir' in data_config:
            self.root_dir = data_config["root_dir"]
        else:
            self.root_dir = "../data/"

        if 'use_data_aug' in data_config:
            self.use_data_aug = data_config['use_data_aug']
        else:
            self.use_data_aug = False

        if 'input_dim' in data_config:
            self.input_resolution = data_config['input_dim']
        else:
            self.input_resolution = 256

        # ----- transform
        self.transform = transforms.Compose(
                    [Rescale((self.input_resolution, self.input_resolution)), ToTensor()])
        self.aug_transform = None
        if self.use_data_aug:
            self.aug_transform = transforms.Compose(
                        [ColorJitter(brightness=0.1, hue=0.05, saturation=0.03),
                         RandomHorizontalFlip(),
                         Rescale((256, 256)),
                         RandomCrop(size=self.input_resolution, padding=int(self.input_resolution * 0.05), padding_mode = 'edge'),
                         ToTensor()])

        # ----- init data list
        self.init_data_list()


    def get_data_from_dir(self, data_dir, split = 'train'):
        data_list = []
        self.data_dict[split] = {}

        groups_l = os.listdir(data_dir)
        groups_l = [group_i for group_i in groups_l if group_i.isnumeric()]

        data_iter = 0
        for group_i in groups_l:
            self.data_dict[split][str(group_i)] = {}
            self.data_dict[split][str(group_i)]['img_list'] = []
            data_dir_i = os.path.join(data_dir, str(group_i))

            imgs_l = os.listdir(data_dir_i)
            imgs_l = [img_i for img_i in imgs_l if 'jpg' in img_i]

            for img_i in imgs_l:
                self.data_dict[split][data_iter] = {}
                self.data_dict[split][data_iter]['img_name'] = img_i
                self.data_dict[split][data_iter]['img_dir'] = os.path.join(data_dir_i, img_i)
                self.data_dict[split][data_iter]['label_no'] = group_i
                self.data_dict[split][str(group_i)]['img_list'].append(img_i)
                data_list.append(data_iter)
                data_iter += 1

        # print distribution
        for group_i in groups_l:
            print("{}: class {} has {} imgs".format(split, str(group_i), len(self.data_dict[split][str(group_i)]['img_list'])))

        return data_list


    def init_data_list(self):
        self.train_root_dir = os.path.join(self.root_dir, 'train')
        self.test_root_dir = os.path.join(self.root_dir, 'test')

        self.data_dict = {}
        self.train_data_list = self.get_data_from_dir(self.train_root_dir, split = 'train')

        self.test_data_list = self.get_data_from_dir(self.test_root_dir, split = 'test')

        if self.split == 'train':
            self.data_list = self.train_data_list

        if self.split ==  'split':
            self.data_list = self.test_data_list


    def train(self):
        self.data_list = self.train_data_list


    def test(self):
        self.data_list = self.test_data_list


    def get_num_of_classes(self):
        return len(self.data_dict[self.split])


    def __len__(self):
        return len(self.data_list)


    def get_data(self, idx):
        sample = {}

        data_dict_i = self.data_dict[self.split][idx]
        img_dir = data_dict_i['img_dir']
        image = io.imread(img_dir)[:, :, :3]
        label = data_dict_i['label_no']
        sample['image'] = image
        sample['label'] = label

        return sample


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.get_data(idx)

        if self.aug_transform is not None:
            sample = self.aug_transform(sample)
        else:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    debug_iteration = True

    data_config = {}
    data_config['split'] = 'train'
    data_config['root_dir'] = "/Users/kaijia/PycharmProjects/Pytorch_beginner/data/"
    data_config['use_data_aug'] = False
    data_config['input_dim'] = 256


    custom_data = CustomDataset(data_config=data_config)

    if debug_iteration:
        # way 1 directly use index
        # data_i = custom_data[0]

        dataloader = DataLoader(custom_data, batch_size=4,
                                shuffle=False, num_workers=0)

        # way 2 use dataloader
        # a) use next
        if False:
            examples = enumerate(dataloader)
            batch_idx, sample_batched = next(examples)

        # b) use for
        for i_batch, sample_batched in enumerate(dataloader):
            show_landmarks_batch(sample_batched)

