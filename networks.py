
import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, dropRate):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=dropRate),
            nn.Flatten(),
            nn.Linear(8736, 32))

    def forward(self, x):
        x = x.unsqueeze(1)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        return output

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)



