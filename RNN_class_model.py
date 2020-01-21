#simple rnn for classification
from torch.autograd import Variable
from nxlearn.ml.modules import RBLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
class Sample_lstm(nn.Module):
    def __init__(self, num_tags:int, batch_size):
        super().__init__()
        self.lstm = RBLSTM(143, 128, 1, p=0.02) #lstm model created by us, generic rnn can be used
        self.hidden = nn.Linear(256,80)
        self.hidden2 = nn.Linear(80,num_tags)
    def forward(self,features,seq_len):
        outputs = self.lstm(features[:,:,:143],seq_len)
        sequence1 = self.hidden(F.relu(outputs))
        output = self.hidden2(F.relu(sequence1))
        return output