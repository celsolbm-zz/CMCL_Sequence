
class loss_v1(nn.Module):
    """
    Loss function version 1
    Using stochastic labeling
    In this case we have a sequence type data and we want to classify each element on the sequence
    the batch size therefore will be the sequence len of each sequence
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
        closs = nn.CrossEntropyLoss(reduction='none') #different from tensorflow, this one uses log_softmax, but results are same
        closs_list = [closs(logits.squeeze(1), labels.squeeze(1)) for logits in logits_list] 
        beta = 0.75 #tunable parameter
        a = beta #was too lazy to look for all the "a"s in the code 
        softmax_list = [torch.clamp(F.softmax(logits.squeeze(1),dim=1),1e-10,1.0) for logits in logits_list] #list of softmax, len of M and each element has dimmension num_class
        entropy_list = [-torch.log(torch.tensor(num_class).float())-torch.mean(torch.log(softmax)) for softmax in softmax_list]
        loss_list = []
        for m in range(self.NUM_MODELS):
            loss_list.append( closs_list[m] + a*sum(entropy_list[:m] + entropy_list[m+1:])  )
        temp, min_index = torch.topk(-torch.tensor(list(map(list, zip(*loss_list)))),self.k) #check which models will be used for each data label based on their loss performance
        min_index = torch.transpose(min_index,0,1)
        random_labels = torch.randint(0,10, (self.NUM_MODELS, seq_len)) #create the random labels
        for m in range(self.NUM_MODELS):
            total_condition = torch.tensor([False]*int(seq_len)) #create the mask that is going to be used on the new loss calc
            for topk in range(self.k):
                condition = torch.eq(min_index[topk], m) #create the mask to be used on the labeling selection
                total_condition = total_condition + condition
                if topk == 0:
                    new_labels = torch.where(condition.cuda(), labels.squeeze(1), random_labels[m].cuda())
                else:
                    new_labels = torch.where(condition.cuda(), labels.squeeze(1), new_labels)
            #calculate the new classification loss with the labels correctly sampled and changed
            classification_loss = torch.where(total_condition.cuda(),torch.ones(int(seq_len)).cuda(),torch.tensor([a]*int(seq_len)).cuda())*closs(logits_list[m].squeeze(1), new_labels)
            total_loss += torch.mean(classification_loss)
        return total_loss