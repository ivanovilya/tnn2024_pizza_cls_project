import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def get_model(model_config):
    if model_config.architecture == 'mamedov_model':
        model = ConvNet(num_classes=model_config.num_classes)
    elif model_config.architecture == 'Askarkhujaev_model':
        model = Askarkhujaev_Network(num_classes=model_config.num_classes)
    elif model_config.architecture == 'Sidorchuk_model':
        model = SidorchukNetwork()
    elif model_config.architecture == 'polonskaya_model':
        model = ResNet18Model(num_classes=model_config.num_classes)
    elif model_config.architecture == 'ashrapov_model':
        return AshrapovNetwork(num_classes=model_config.num_classes, dropout=model_config.dropout)
    elif model_config.architecture == 'ruzmetov_model':
        return RuzmetovNetwork(num_classes=model_config.num_classes)
    return model


class ConvNet(nn.Module):
    def __init__(self, num_classes=46):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


class Askarkhujaev_Network(nn.Module):
    def __init__(self, num_classes):
        super(Askarkhujaev_Network, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 30 * 30, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SidorchukNetwork(nn.Module):
    def __init__(self):
        super(SidorchukNetwork, self).__init__()

        self.max_pool = nn.MaxPool2d(2)
        self.global_pool = nn.MaxPool2d(2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(60, 60, 3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer14 = nn.Sequential(
            nn.Conv2d(60, 60, 3, stride=2),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer15 = nn.Sequential(
            nn.Conv2d(60, 60, 3, stride=2),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer16 = nn.Sequential(
            nn.Conv2d(60, 60, 3, stride=2),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.layer17 = nn.Sequential(
            nn.Conv2d(60, 60, 3, stride=2),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.linear_layer = torch.nn.Linear(13500, 46)

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        tmp1 = X
        X = self.layer3(X)
        X = self.layer4(X)
        X = X + tmp1
        tmp2 = X
        X = self.layer8(X)
        X = self.layer9(X)
        X = X + tmp2
        tmp3 = X
        X = self.layer10(X)
        X = self.layer11(X)
        X = X + tmp3
        tmp4 = X
        X = self.layer12(X)
        X = self.layer13(X)
        X = X + tmp4
        X = self.layer14(X)
        X = self.layer15(X)
        X = self.layer15(X)
        X = self.layer17(X)

        X = torch.flatten(X, 1)
        X = self.linear_layer(X)
        # X = torch.softmax(X, dim=1)

        return X


class AshrapovNetwork(nn.Module):
    def __init__(self, num_classes=46, dropout=0.5):
        super(AshrapovNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),

            nn.Flatten(),
            nn.Linear(65536, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class Averin_Network(nn.Module):
    def __init__(self, num_classes=46, dropout=0.4):
        super(Averin_Network, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.tail(x)
        x = self.classifier(x)
        return x


class RuzmetovNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.layer1_bn1 = nn.BatchNorm2d(64)
        self.layer1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_bn2 = nn.BatchNorm2d(64)
        self.layer1_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer1_bn3 = nn.BatchNorm2d(256)
        self.layer1_downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer2_conv1 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.layer2_bn1 = nn.BatchNorm2d(128)
        self.layer2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_bn2 = nn.BatchNorm2d(128)
        self.layer2_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_bn3 = nn.BatchNorm2d(512)
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.layer3_conv1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.layer3_bn1 = nn.BatchNorm2d(256)
        self.layer3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_bn2 = nn.BatchNorm2d(256)
        self.layer3_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_bn3 = nn.BatchNorm2d(1024)
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024)
        )

        # Layer 4
        self.layer4_conv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.layer4_bn1 = nn.BatchNorm2d(512)
        self.layer4_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_bn2 = nn.BatchNorm2d(512)
        self.layer4_conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.layer4_bn3 = nn.BatchNorm2d(2048)
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity = x
        out = self.layer1_conv1(x)
        out = self.layer1_bn1(out)
        out = self.relu(out)
        out = self.layer1_conv2(out)
        out = self.layer1_bn2(out)
        out = self.relu(out)
        out = self.layer1_conv3(out)
        out = self.layer1_bn3(out)
        identity = self.layer1_downsample(identity)
        out += identity
        x = self.relu(out)

        identity = x
        out = self.layer2_conv1(x)
        out = self.layer2_bn1(out)
        out = self.relu(out)
        out = self.layer2_conv2(out)
        out = self.layer2_bn2(out)
        out = self.relu(out)
        out = self.layer2_conv3(out)
        out = self.layer2_bn3(out)
        identity = self.layer2_downsample(identity)
        out += identity
        x = self.relu(out)

        identity = x
        out = self.layer3_conv1(x)
        out = self.layer3_bn1(out)
        out = self.relu(out)
        out = self.layer3_conv2(out)
        out = self.layer3_bn2(out)
        out = self.relu(out)
        out = self.layer3_conv3(out)
        out = self.layer3_bn3(out)
        identity = self.layer3_downsample(identity)
        out += identity
        x = self.relu(out)

        identity = x
        out = self.layer4_conv1(x)
        out = self.layer4_bn1(out)
        out = self.relu(out)
        out = self.layer4_conv2(out)
        out = self.layer4_bn2(out)
        out = self.relu(out)
        out = self.layer4_conv3(out)
        out = self.layer4_bn3(out)
        identity = self.layer4_downsample(identity)
        out += identity
        x = self.relu(out)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
