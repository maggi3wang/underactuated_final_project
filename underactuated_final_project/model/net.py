import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)

        self.fc1 = nn.Linear(512+84, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 8)

    def forward(self, image, joint_descriptor):
        x1 = self.resnet(image)
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = joint_descriptor
        x2 = F.normalize(x2, p=2, dim=1)
        
        x = torch.cat((x1, x2), dim=1)
        x = F.normalize(x, p=2, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x