import RNN_class_model
import torch
import torch.nn as nn
import torch.nn.functional as F

class ensemble_model(nn.Module):
    def __init__(self, num_models:int, k:int, batch_size:int):
        super().__init__()
        self.num_models = num_models
        self.k = k
        #cannot use for loops because model.parameters() or the optimizer doesnt work
        #I know its ugly, I will make it prettier later
        self.model1 = Sample_lstm(10,batch_size).cuda()
        self.model2 = Sample_lstm(10,batch_size).cuda()
        self.model3 = Sample_lstm(10,batch_size).cuda()
        self.model4 = Sample_lstm(10,batch_size).cuda()
        self.model5 = Sample_lstm(10,batch_size).cuda()
        self.model6 = Sample_lstm(10,batch_size).cuda()
        self.model7 = Sample_lstm(10,batch_size).cuda()
        self.model8 = Sample_lstm(10,batch_size).cuda()
        self.model9 = Sample_lstm(10,batch_size).cuda()
        self.model10 = Sample_lstm(10,batch_size).cuda()
        self.model11 = Sample_lstm(10,batch_size).cuda()
        self.model12 = Sample_lstm(10,batch_size).cuda()
        self.model13 = Sample_lstm(10,batch_size).cuda()
        self.model14 = Sample_lstm(10,batch_size).cuda()
        self.model15 = Sample_lstm(10,batch_size).cuda()
        self.model16 = Sample_lstm(10,batch_size).cuda()
        self.model17 = Sample_lstm(10,batch_size).cuda()
        self.model18 = Sample_lstm(10,batch_size).cuda()
        self.model19 = Sample_lstm(10,batch_size).cuda()
        self.model20 = Sample_lstm(10,batch_size).cuda()
        #for i in range(num_models):
        #    self.models.append(Sample_lstm(10,batch_size).cuda())
    def forward(self,features, seq_len):
        ret = []
        ret.append(self.model1(features,seq_len))
        ret.append(self.model2(features,seq_len))
        ret.append(self.model3(features,seq_len))
        ret.append(self.model4(features,seq_len))
        ret.append(self.model5(features,seq_len))
        ret.append(self.model6(features,seq_len))
        ret.append(self.model7(features,seq_len))
        ret.append(self.model8(features,seq_len))
        ret.append(self.model9(features,seq_len))
        ret.append(self.model10(features,seq_len))
        ret.append(self.model11(features,seq_len))
        ret.append(self.model12(features,seq_len))
        ret.append(self.model13(features,seq_len))
        ret.append(self.model14(features,seq_len))
        ret.append(self.model15(features,seq_len))
        ret.append(self.model16(features,seq_len))
        ret.append(self.model17(features,seq_len))
        ret.append(self.model18(features,seq_len))
        ret.append(self.model19(features,seq_len))
        ret.append(self.model20(features,seq_len))
        return ret
        #for i in range(self.num_models):
        #    ret.append(self.models[i](features,seq_len))
        #return ret