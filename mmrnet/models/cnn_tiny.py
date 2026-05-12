import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Tiny(nn.Module):
    def __init__(self, info=None):
        super(CNN_Tiny, self).__init__()
        self.num_classes = info.get('num_classes')

        # use additional features as channels
        input_channels = info.get('num_features') - 3 + 1 # (xyz are not used as channels, +1 because there is 1 baseline channel)
        self.conv1a = nn.Conv3d(input_channels, 33, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1a = nn.BatchNorm3d(33)
        self.conv1b = nn.Conv3d(33, 33, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm3d(33)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv2a = nn.Conv3d(33, 33, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2a = nn.BatchNorm3d(33)
        self.conv2b = nn.Conv3d(33, 33, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2b = nn.BatchNorm3d(33)
        self.pool2 = nn.MaxPool3d(kernel_size=4, stride=4, padding=0)

        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(33*1*4*4, self.num_classes, bias=True)


    def forward(self, x):
        # Input shape: (batch_size, seq_len, D, H, W, C) # (B,T,Z,Y,X,C)
        batch_size, seq_len, D, H, W, C = x.shape

        # Reshape for Conv3D: (batch_size * seq_len, C, D, H, W)
        x = x.permute(0, 1, 5, 2, 3, 4).contiguous().view(-1, C, D, H, W)

        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        x = x.view(batch_size,-1)

        # Apply dropout
        x = self.dropout1(x)

        # Final fully connected layer
        x = self.fc(x)

        return x