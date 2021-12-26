import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import to_variable, log_gaussian_loss
from optimizer import Adam_Langevin


class MCDrop_MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, prop=0.3,
                 init_log_noise=0, classification=False):
        super(MCDrop_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prop = prop

        self.layer1 = nn.Linear(input_dim, 600)
        self.layer2 = nn.Linear(600, 500)
        self.layer3 = nn.Linear(500, 400)
        self.layer4 = nn.Linear(400, 300)
        self.layer5 = nn.Linear(300, 200)
        self.layer6 = nn.Linear(200, 100)
        self.layer7 = nn.Linear(100, 50)
        self.layer8 = nn.Linear(50, output_dim)     
        
        self.activation = nn.ReLU(inplace = True)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))
        self.classification = classification
        
    
    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        x = self.layer1(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer2(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer3(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer4(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer5(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer6(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer7(x)
        x = F.dropout(x, p=self.prop, training=True)
        x = self.activation(x)
        x = self.layer8(x)
        x = F.softmax(x, dim=1)
        if(self.classification):
            x = F.softmax(x, dim=1)
        
        return x


class MCDrop_model:
    def __init__(self, input_dim=1*28*28, output_dim=10, N_train=60000,
                 classification=True, learn_rate=1e-2, init_log_noise=0,
                 batch_size=128, prop=0.3, weight_decay=5e-7):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_decay = weight_decay
        self.N_train = N_train
        self.classification = classification
        self.prop = prop
        
        self.network = MCDrop_MLP(input_dim = self.input_dim, output_dim = self.output_dim, prop=self.prop,
                                init_log_noise = init_log_noise, classification = self.classification)
        self.network.cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=self.learn_rate, weight_decay=self.weight_decay)
    
    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)
        self.optimizer.zero_grad()
        
        output = self.network(x)
        if(self.classification):
            loss = F.cross_entropy(output, y, reduction='sum')
        else:
            #print(output[:,0].shape, y.shape)
            loss = log_gaussian_loss(output[:,0], y[:,0], torch.exp(self.network.log_noise), 1)/len(x) #torch.ones(1).cuda()
            #loss = torch.nn.functional.mse_loss(output[:,0], y)
        
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, num_epochs=64, 
              trainloader=torch.utils.data.dataloader.DataLoader):

        err_train = np.zeros(num_epochs)
        for i in range(num_epochs):
            nb_samples = 0
            for x, y in trainloader:
                err = self.fit(x, y)
                err_train[i] += err
                nb_samples += len(x)

            err_train[i] /= nb_samples

            print('Epoch:', i, 'Train loss = ',  err_train[i])
