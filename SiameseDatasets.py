import numpy as np

from torch.utils.data import Dataset
import torch


class TransferDataset(Dataset):
    def __init__(self, train_User, Defult_User, UserID, epoch):
        self.train_User = train_User
        self.Defult_User = Defult_User
        self.UserID = UserID
        self.train_User_samples = self.train_User.samples
        self.train_User_labels = self.train_User.labels
        self.Defult_User_samples = self.Defult_User.samples
        self.Defult_User_labels = self.Defult_User.labels
        self.epoch = epoch

    def __getitem__(self, index):
        if index % 2 == 0:
            target = 1
        else:
            target = 0

        sample1, label1 = self.train_User_samples[index], self.train_User_labels[index].item()

        if target == 1:

            torch.manual_seed(self.epoch+index)
            torch.cuda.manual_seed(self.epoch+index)
            sample2 = self.train_User_samples[torch.randint(0, self.train_User_samples.size(0), (1,))]

        else:
            torch.manual_seed(self.epoch+index)
            torch.cuda.manual_seed(self.epoch+index)
            sample2 = self.Defult_User_samples[torch.randint(0, self.Defult_User_samples.size(0), (1,))]
        sample2 = torch.squeeze(sample2)
        return (sample1, sample2), target


    def __len__(self):
        return len(self.train_User)

