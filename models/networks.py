import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class TransferNet(nn.Module):
    # TODO: generalize representation of multiple parallel fc layers
    def __init__(self, num_classes=4):
        super(TransferNet, self).__init__()
        # Pretrained part
        resnet = torchvision.models.resnet18(pretrained=True)
        # Dropping the last fc layer
        self.conv = nn.Sequential(*(list(resnet.children())[:-1]))
        """ Uncomment this to freeze earlier layers
        for param in self.conv.parameters():
            param.requires_grad = False
        """
        # Customized part
        feature_size = resnet.fc.in_features
        self.fc = nn.Linear(feature_size, num_classes)


    def forward(self, x):
        x = self.conv(x)
        # Resnet has an average pooling layer before fc layer
        # therefore need to reshape before connecting to fc layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)