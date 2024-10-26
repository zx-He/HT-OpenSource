import torch
import numpy as np
from torch.utils.data import DataLoader

from SiameseDatasets import TransferDataset


def fit(train_User, Defult_User, UserID, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):


    trainLoss_list = []
    for epoch in range(0, start_epoch):
        scheduler.step()
    batchSize = 8


    for epoch in range(start_epoch, n_epochs):
        scheduler.step()


        TransferTrain = TransferDataset(train_User, Defult_User, UserID, epoch)
        cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        TransferTrainset_loader = DataLoader(TransferTrain, batch_size=batchSize, shuffle=False, **kwargs)

        train_loss, metrics = train_epoch(TransferTrainset_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, epoch)
        trainLoss_list.append(train_loss)
    trainloss = np.sum(trainLoss_list)/len(trainLoss_list)

    return trainloss


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, epoch):

    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())


        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

            losses = []
    total_loss /= (batch_idx + 1)
    return total_loss, metrics

