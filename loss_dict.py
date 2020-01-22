class loss_v2(nn.Module):
    """
    Loss function version 2
    Using stochastic labeling
    This is the same as loss_v1 but in here we choose the models to learn each class manually
    this avoid the problem of having K models learn every label
    Args:
        logits_list = list of logits calculated by the model ensemble
        labels = Label input corresponding to the calculated batch
        seq_len = length of the sequence

    """
    def __init__(self, k:int,num_models:int, num_class:int):
        super().__init__()
        self.NUM_MODELS = num_models #number of models in the ensemble
        self.k = k #overlapping parameter
        self.num_class = num_class #
    def forward(self, logits_list,labels,seq_len):
        num_class = 10 #for our case its 10, you can change this depending on the database
        total_loss = 0
        learn = {
                    0:(0,1),
                    1:(2,3),
                    2:(4,5),
                    3:(6,7),
                    4:(8,9),
                    5:(10,11),
                    6:(12,13),
                    7:(14,15),
                    8:(16,17),
                    9:(18,19)
                }
        closs = nn.CrossEntropyLoss(reduction='none') #different from tensorflow, this one uses log_softmax, but results are same
        a = 0.75 #beta, tunable parameter
        index = []
        for i in labels.squeeze(1):
            index.append(list(learn[int(i)]))
        min_index = torch.tensor(index)
        min_index = torch.transpose(min_index,0,1)
        #print("min_index = ", min_index)
        random_labels = torch.randint(0,10, (self.NUM_MODELS, seq_len)) #create the random labels
        for m in range(self.NUM_MODELS):
            total_condition = torch.tensor([False]*int(seq_len)) #create the mask that is going to be used on the new loss calc
            for topk in range(self.k):
                condition = torch.eq(min_index[topk], m) #create the mask to be used on the labeling selection
                total_condition = total_condition + condition
                #ind2 = 0
                #for i in labels.squeeze(1):
                #    if i==3:
                #        condition[ind2] = False
                #    ind2+=1
                if topk == 0:
                    new_labels = torch.where(condition.cuda(), labels.squeeze(1), random_labels[m].cuda())
                else:
                    new_labels = torch.where(condition.cuda(), labels.squeeze(1), new_labels)
            #calculate the new classification loss with the labels correctly sampled and changed
            classification_loss = torch.where(total_condition.cuda(),torch.ones(int(seq_len)).cuda(),torch.tensor([a]*int(seq_len)).cuda())*closs(logits_list[m].squeeze(1), new_labels)
            total_loss += torch.mean(classification_loss)
        return total_loss