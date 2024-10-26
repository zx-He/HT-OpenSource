import os
import torch
from torch.utils.data import Dataset, Subset
import numpy as np

def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = np.array([list(map(float, line.strip().split(','))) for line in lines])
    return data

class getOriginDataset(Dataset):
    def __init__(self, root_dir, train=True, userList = [], dataSession = []):
        self.root_dir = root_dir
        self.samples = []
        self.labels = []
        self.train = train
        self.userList = userList
        self.dataSession = dataSession
        for user in userList:
            UserPath = os.path.join(root_dir, str(user))
            for session in dataSession:
                dataSession_path = os.path.join(UserPath, str(session))
                for j in range(1, 11):
                    file_path = os.path.join(dataSession_path, str(j) + ".txt")
                    data = read_data(file_path)
                    data = data.flatten()
                    self.samples.append(data)
                    self.labels.append(user)

        self.samples = torch.tensor(self.samples).float()
        self.labels = torch.tensor(self.labels).float()


    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label

    def __len__(self):
        return len(self.samples)



class GetSubset(Subset):
    def __init__(self, dataset, indices, train=True, transform=None):
        super().__init__(dataset, indices)
        self.train = train
        self.transform = transform
        self.samples = dataset.samples[indices]
        self.labels = dataset.labels[indices]

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return torch.tensor(sample), torch.tensor(label)

    def __len__(self):
        return len(self.samples)



