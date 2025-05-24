import glob
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from torch.utils.data import Dataset
import torch
from model import lstm, Generator
from Config import dic_obj as opt


class Dataset_m(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data_file_list = sorted(glob.glob(f'{self.root}/2017/*.jpg'))
        pass

    def __getitem__(self, index):
        cur_len = 0
        print("index", index)
        while cur_len <= opt.J:
            # self.sel_data_list = sorted(glob.glob(f'self.data_file_list/*.jpg'))
            cur_len = len(self.data_file_list)
            index += 1
        data_idx = int(np.random.choice(range(0, cur_len-opt.J), 1))
        while data_idx+opt.J >= len(self.data_file_list):
            data_idx = data_idx - 1
        sel_data_list = []
        for i in range(opt.J + 1):
            data = Image.open(self.data_file_list[data_idx + i])
            data = self.transform(data)
            sel_data_list.append(data)
        data_mat = torch.cat(sel_data_list, dim=0)
        assert len(sel_data_list) == opt.J + 1, 'error ！'
        return data_mat[:opt.J, :, :], data_mat[-1, :, :]

    def __len__(self):
        return len(self.data_file_list)


if __name__ == '__main__':
    transforms_train = transforms.Compose([transforms.Resize((opt.img_size, opt.img_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=0.5, std=0.5),
                                           ])

    my_dataset = Dataset_m('./my_data/zaolansu', transform=transforms_train)
    dataloader = DataLoader(
        dataset=my_dataset,
        batch_size=opt.batch_size_lstm,
        shuffle=True,
        drop_last=True,
    )
    iteration = 0
    for epoch in range(opt.epochs_lstm):
        for i, (imgs, target) in enumerate(dataloader):
            imgs = imgs.cuda()
            imgs = imgs*0.5+0.5
            imgs = torch.clamp(imgs, 0, 255)
            # imgs = torchvision.transforms.ToPILImage()(imgs)
            iteration += 1
            if iteration % 1 == 0:
                for i in range(opt.J):
                    save_image(imgs.data[0][i:i + 1].reshape(opt.img_size, opt.img_size),
                               f"images_lstm/E{iteration+1}_aseries_{i}.png", nrow=4, normalize=False)