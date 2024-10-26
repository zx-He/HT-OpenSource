
import torch
import random
import torch.optim as optim
import torch.nn as nn
import os

from OriginDataSet import getOriginDataset, GetSubset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from losses import ContrastiveLoss
from networks import SiameseNet
from trainer import fit


def count_samples_above_threshold(lst, t):
    count = 0
    for sample in lst:
        if sample > t:
            count += 1
    return count

if __name__ == '__main__':

    seed = 10000
    lr = 0.001
    n_epochs = 45
    batchSize = 8

    distance_le = []
    distance_ile = []
    trainSampleNum_list = [10]
    userList = [1, 2, 4, 6, 7, 9, 10, 12, 14, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    default_userList = [3, 5, 8, 11, 13, 15, 16, 17, 18, 24]
    dataSession = [1,2,3,4,5]
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    for trainSampleNum in trainSampleNum_list:

        for i in userList:
            embedding_net = torch.load("pre-trained_net.pth")
            embedding_net.train()
            for name, module in embedding_net.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0

            UserID = i
            distance_le_curr = []
            distance_ile_curr = []


            Default_User = getOriginDataset('/HT-Dataset/',
                                       userList=default_userList, dataSession=dataSession)

            User = getOriginDataset('/HT-Dataset/',
                                           userList=[UserID], dataSession=dataSession)
            Illegal_userList = userList[:]
            if UserID in Illegal_userList:
                Illegal_userList.remove(i)
            Illegal_User = getOriginDataset('/HT-Dataset/', userList=Illegal_userList, dataSession=dataSession)

            train_indices = []
            test_indices = []
            indices = [j for j, label in enumerate(User.labels) if label == UserID]
            random.seed(seed)
            train_indices.extend(random.sample(indices, trainSampleNum))


            test_indices.extend([index for index in indices if index not in train_indices])
            train_User = GetSubset(User, train_indices, train=True)
            test_User = GetSubset(User, test_indices, train=False)

            cuda = torch.cuda.is_available()
            margin = 1.
            model = SiameseNet(embedding_net)
            if cuda:
                model.cuda()

            loss_fn = ContrastiveLoss(margin)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
            log_interval = 55
            trainloss = fit(train_User, Default_User, UserID, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


            temp_dataloader = DataLoader(train_User, batch_size=1)
            output_list = []
            with torch.no_grad():
                for data in temp_dataloader:
                    sample, target = data
                    sample = sample.cuda()

                    output = embedding_net(sample)
                    output_list.append(output)
            concatenated = torch.cat(output_list, dim=0)
            summed = torch.sum(concatenated, dim=0)
            template = summed / len(output_list)
            template = template.unsqueeze(0)

            template_cpu = template.cpu()
            template_np = template_cpu.numpy()


            Illegal_dataloader = DataLoader(Illegal_User, batch_size=1)
            legal_dataloader = DataLoader(test_User, batch_size=1)
            embedding_net.eval()


            with torch.no_grad():
                for data in legal_dataloader:
                    samples, label = data
                    samples = samples.cuda()
                    outputs = embedding_net(samples)
                    distance = torch.dist(template, outputs)
                    distance_le_curr.append(distance)
                    distance_le.append(distance)

            with torch.no_grad():
                for data in Illegal_dataloader:
                    samples, label = data
                    samples = samples.cuda()
                    outputs = embedding_net(samples)
                    distance = torch.dist(template, outputs)
                    distance_ile_curr.append(distance)
                    distance_ile.append(distance)

            thresh = 0.49
            FN = count_samples_above_threshold(distance_le_curr, thresh)
            TP = len(distance_le_curr) - FN
            TN = count_samples_above_threshold(distance_ile_curr, thresh)
            FP = len(distance_ile_curr) - TN
            FAR = FP / (FP + TN)
            FRR = FN / (TP + FN)
            TPR = TP / (TP + FN)
            TNR = TN / (TN + FP)
            BAC = 1 / 2 * (TPR + TNR)
            message = '\ntrainSampleNum:{}, batchSize: {}, lr: {}, n_epochs: {}, thresh: {}, seed: {}, FAR: {}, FRR: {}, BAC: {}'.format(
                trainSampleNum, batchSize, lr, n_epochs, thresh, seed, FAR, FRR, BAC)
            print(message)

        thresh = 0.49

        FN = count_samples_above_threshold(distance_le, thresh)
        TP = len(distance_le) - FN
        TN = count_samples_above_threshold(distance_ile, thresh)
        FP = len(distance_ile) - TN
        FAR = FP /(FP+TN)
        FRR = FN /(TP+FN)
        TPR = TP /(TP+FN)
        TNR = TN /(TN+FP)
        BAC = 1/2 * (TPR + TNR)

        message = '\ntrainSampleNum:{}, batchSize: {}, lr: {}, n_epochs: {}, thresh: {}, seed: {}, FAR: {}, FRR: {}, BAC: {}'.format(trainSampleNum, batchSize, lr, n_epochs, thresh, seed, FAR, FRR, BAC)
        print(message)







