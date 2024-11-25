import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MultiLeNetR(nn.Module):
    def __init__(self):
        super(MultiLeNetR, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc = nn.Linear(320, 50)

    def dropout2dwithmask(self, x, mask):
        channel_size = x.shape[1]
        if mask is None:
            mask = Variable(torch.bernoulli(torch.ones(1,channel_size,1,1)*0.5).cuda())
        mask = mask.expand(x.shape)
        return mask

    def forward(self, x):
        mask = None
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2(x)
        mask = self.dropout2dwithmask(x, mask).to(x.device)
        # self.training: a parameter in nn.Module
        if self.training:
            x = x*mask
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc(x))
        return x
        # LResult,_ = self.Lpred(x,None)
        # RResult,_ = self.Rpred(x,None)
        # return [LResult,RResult]

