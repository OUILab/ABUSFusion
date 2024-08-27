import torch
import torch.nn as nn
import torch.nn.functional as F


class DCLNet(nn.Module):
    def __init__(self, num_input_frames=5, num_classes=6):
        super(DCLNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4)
        self.layer3 = self._make_layer(256, 512, 6)
        self.layer4 = self._make_layer(512, 1024, 3)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(1024, num_classes)

        self.attention = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(512, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(
                nn.Conv3d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            )
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        attention = self.attention(x)
        x = x * attention

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
