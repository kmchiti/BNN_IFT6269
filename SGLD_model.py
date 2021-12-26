import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Utils import Unif_Layer, to_variable, log_gaussian_loss
from optimizer import Adam_Langevin


class SGLD_MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1,
                 init_log_noise=0, classification=False):
        super(SGLD_MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer1 = Unif_Layer(input_dim, 600)
        self.layer2 = Unif_Layer(600, 500)
        self.layer3 = Unif_Layer(500, 400)
        self.layer4 = Unif_Layer(400, 300)
        self.layer5 = Unif_Layer(300, 200)
        self.layer6 = Unif_Layer(200, 100)
        self.layer7 = Unif_Layer(100, 50)
        self.layer8 = Unif_Layer(50, output_dim)     
        
        self.activation = nn.ReLU(inplace = True)
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([init_log_noise]))
        self.classification = classification
        
    
    def forward(self, x):
        
        x = x.view(-1, self.input_dim)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.activation(x)
        x = self.layer4(x)
        x = self.activation(x)
        x = self.layer5(x)
        x = self.activation(x)
        x = self.layer6(x)
        x = self.activation(x)
        x = self.layer7(x)
        x = self.activation(x)
        x = self.layer8(x)
        if(self.classification):
            x = F.softmax(x, dim=1)
        
        return x


class SGLD_model:
    def __init__(self, input_dim=1*28*28, output_dim=10, N_train=60000,
                 classification=True, learn_rate=1e-2, init_log_noise=0,
                 batch_size=128, weight_decay=5e-7):
        
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_decay = weight_decay
        self.N_train = N_train
        self.classification = classification
        
        self.network = SGLD_MLP(input_dim = self.input_dim, output_dim = self.output_dim,
                                init_log_noise = init_log_noise, classification = self.classification)
        self.network.cuda()
        self.optimizer = Adam_Langevin(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
    
    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=True)
        self.optimizer.zero_grad()
        
        output = self.network(x)
        if(self.classification):
            loss = F.cross_entropy(output, y, reduction='mean')
            loss = loss * self.N_train 
        else:
            #print(output[:,0].shape, y.shape)
            loss = log_gaussian_loss(output[:,0], y[:,0], torch.exp(self.network.log_noise), 1)/len(x) #torch.ones(1).cuda()
            #loss = torch.nn.functional.mse_loss(output[:,0], y)
        
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, num_nets=50, mix_epochs=2, burnin_epochs=32,
              gamma=-.65, b=.9, a=.01, 
              trainloader=torch.utils.data.dataloader.DataLoader):
        num_epochs = mix_epochs*num_nets + burnin_epochs
        nets, losses = [], []
        pred_cost_train = np.zeros(num_epochs)
        err_train = np.zeros(num_epochs)
        best_err = np.inf

        for i in range(num_epochs):
            nb_samples = 0
            self.learn_rate = a*(b+i)**gamma
            for x, y in trainloader:
                err = self.fit(x, y)
                err_train[i] += err
                nb_samples += len(x)

            err_train[i] /= nb_samples

            if(i%mix_epochs==0):
              print('Epoch:', i, 'Train loss = ',  err_train[i])
            if i % mix_epochs == 0 and i > burnin_epochs: nets.append(copy.deepcopy(self.network))
        
        return nets
