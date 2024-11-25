import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class MultiLeNetO(nn.Module):
    def __init__(self):
        super(MultiLeNetO, self).__init__()
        self.fc1 = nn.Linear(50, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mask = Variable(torch.bernoulli(x.data.new(x.data.size()).fill_(0.5)))
        if self.training:
            x = x*mask
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)