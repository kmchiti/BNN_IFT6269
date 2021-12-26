import time
import copy
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from scipy.stats import entropy
from tqdm import tqdm
sns.set()

class Unif_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Unif_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
        self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        
    def forward(self, x):
        return torch.mm(x, self.weights) + self.biases

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    return - (log_coeff + exponent).sum()

def butter_lowpass_filter(data, cutoff, fs, order, nyq):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def prep_data_regression(path_data='/content/SolarPrediction.csv'):
    data = pd.read_csv(path_data)
    y = np.array(data['Radiation'])
    x = np.arange(len(y))
    y_new = []
    x_new = []
    cut = 144
    for i in range(1,len(y)):
      if(i%cut == 0):
        y_new.append(np.mean(y[i-cut:i]))
        x_new.append(i//cut)
    n = len(x_new) 
    fs = 20.0     
    T = n/fs      
    cutoff = 2   
    nyq = 0.5 * fs  
    order = 2      

    y2 = butter_lowpass_filter(y_new, cutoff, fs, order, nyq)
    x_new = np.array(x_new)
    x_new = (x_new-x_new.mean())/70
    y2 = np.log(y2)*3 - 15

    mask = np.concatenate((np.arange(0,72), np.arange(90,143), np.arange(158,len(x_new))), 0)
    x_train = x_new[mask]
    y_train = y2[mask]
    x_test = x_new
    y_test = y2
    return x_train, y_train, x_test, y_test



def prep_data_classification():
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    use_cuda = torch.cuda.is_available()

    trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
    valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
    return trainset, valset



def data_loader(classification):
    if(classification):
        # train config
        NTrainPointsMNIST = 60000
        batch_size = 128
        trainset, valset = prep_data_classification()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)

    else:
        x_train, y_train, x_test, y_test = prep_data_regression()
        x_train = torch.tensor(x_train).float()
        y_train = torch.tensor(y_train).float()
        x_test = torch.tensor(x_test).float()
        y_test = torch.tensor(y_test).float()
        
        trainloader = DataLoader(TensorDataset(x_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=len(x_train), shuffle=True)
        valloader = DataLoader(TensorDataset(x_test.unsqueeze(1), y_test.unsqueeze(1)), batch_size=len(x_test), shuffle=True)

    return trainloader, valloader



def test_acc(valloader, nets):
    total_err = 0
    predictions = []
    for x, y in valloader:
        err = 0
        for network in nets:
            x, y = to_variable(var=(x, y.long()), cuda=True)
            probs = network(x)
            #predictions.append(torch.abs(probs - ))
            pred = probs.data.max(dim=1, keepdim=False)[1] 
            err += pred.ne(y.data).sum()
        total_err += 1 - err/len(valloader.dataset)
        return total_err/len(valloader)


def offset_image(coord, name, ax, x_new):
    img = x_new[0,:,:].cpu().data.numpy()
    im = OffsetImage(img, cmap='gray') #zoom=0.72,
    im.image.axes = ax

    ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -34.), frameon=False,
                        xycoords='data',  boxcoords="offset points", pad=0)

    ax.add_artist(ab)


def test_SGLD(x, target, nets):
    pred_list = []
    H_list = []
    for network in nets:
      pred = network.forward(x.float().cuda())
      #pred = F.softmax(pred, dim=1).data
      pred_list.append(pred)
      H_list.append(entropy(pred.cpu().data.numpy(), axis=1))

    pred_list = torch.stack(pred_list)
    pred_mean = pred_list.mean(dim=0)
    #H = -torch.sum(pred_mean*torch.log(pred_mean)).cpu().data.numpy()
    H_mean = entropy(pred_mean.cpu().data.numpy(), axis=1)
    H_var = np.std(H_list, axis=0)

    pred_label_list = torch.argmax(pred_list, dim=2)
    acc = torch.sum(pred_label_list==target)/pred_label_list.shape[0]
    acc = acc.cpu().data.numpy()
    c_ = torch.mode(pred_label_list, dim=0)
    variation_ratio = 1 - torch.sum(pred_label_list==c_.values[0]*torch.ones_like(pred_label_list))/pred_label_list.shape[0]
    variation_ratio = variation_ratio.cpu().data.numpy()

    return H_list, H_mean, H_var, pred_list, acc, variation_ratio

def test_MC(x, target, test_iter, net):
    pred_list = []
    H_list = []
    for i in range(test_iter):
        pred = net.network.forward(x.float().cuda())
        #pred = F.softmax(pred, dim=1).data
        pred_list.append(pred)
        H_list.append(entropy(pred.cpu().data.numpy(), axis=1))

    pred_list = torch.stack(pred_list)
    pred_mean = pred_list.mean(dim=0)
    H_mean = entropy(pred_mean.cpu().data.numpy(), axis=1)
    H_var = np.std(H_list, axis=0)

    pred_label_list = torch.argmax(pred_list, dim=2)
    acc = torch.sum(pred_label_list==target)/pred_label_list.shape[0]
    acc = acc.cpu().data.numpy()
    c_ = torch.mode(pred_label_list, dim=0)
    variation_ratio = 1 - torch.sum(pred_label_list==c_.values[0]*torch.ones_like(pred_label_list))/pred_label_list.shape[0]
    variation_ratio = variation_ratio.cpu().data.numpy()

    return H_list, H_mean, H_var, pred_list, acc, variation_ratio

def plot_Uncertainty(x, target, nets, err_type_):
    degress = np.linspace(0,150, 20)
    noise_list = np.linspace(0,2, 20)
    if(err_type_=="noise"):
        input_list = noise_list
    elif(err_type_=="rotate"):
        input_list = degress
    H_mean_list = []
    H_var_list = []
    acc_list = []
    variation_ratio_list = []
    x_new_list = []
    softmax_means = []
    softmax_vars = []
    for noise in tqdm(input_list):
        if(err_type_=="noise"):
            x_new = torch.rand_like(x)*noise + x
        elif(err_type_=="rotate"):
            x_new = torchvision.transforms.functional.rotate(x, noise)
        x_new_list.append(x_new)
        _, H_mean, H_var, pred_list, acc, variation_ratio = test_SGLD(x_new, target, nets)
        H_mean_list.append(H_mean)
        H_var_list.append(H_var)
        acc_list.append(acc)
        variation_ratio_list.append(variation_ratio)
        softmax_means.append(torch.mean(pred_list[:,0,target], dim=0).cpu().data.numpy())
        softmax_vars.append(torch.std(pred_list[:,0,target], dim=0).cpu().data.numpy())

    H_mean_list = np.array(H_mean_list)
    H_var_list = np.array(H_var_list)
    softmax_means = np.array(softmax_means)
    softmax_vars = np.array(softmax_vars)
    
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(7)
    ax.set_title("SGLD", fontweight="bold")
    if(err_type_ == "noise"):
        ax.set_xlabel("noise")
    elif(err_type_ == "rotate"):
        ax.set_xlabel("rotation degree")
    ax.set_ylabel("nat", color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    lns1 = ax.plot(input_list, H_mean_list, color='#1f77b4', label="Entropy")
    ax.fill_between(input_list, H_mean_list[:,0]-H_var_list[:,0], H_mean_list[:,0]+H_var_list[:,0], alpha=0.3, color='#1f77b4')

    ax2 = ax.twinx()
    ax2.set_ylabel('predictive accuracy', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    lns2 = ax2.plot(input_list, softmax_means, label="softmaxt", color='#ff7f0e')
    ax2.fill_between(input_list, softmax_means-softmax_vars, softmax_means+softmax_vars, alpha=0.3, color='#ff7f0e')
    ax2.grid(None)

    ax.tick_params(axis='x', which='major', pad=26)
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    for i, c in enumerate(input_list):
        if(i%(len(input_list)//5) == 0 or i==len(input_list)-1):
          offset_image(c, c, ax, x_new_list[i])

    if(err_type_=="noise"):
        plt.savefig("Entropy_noise_SGLD.pdf", bbox_inches='tight')
    elif(err_type_=="rotate"):
        plt.savefig("Entropy_rotate_SGLD.pdf", bbox_inches='tight')


        
def plot_Uncertainty_Regression(valloader, trainloader, nets):
    x_test, y_test = valloader.dataset.tensors
    x_train, y_train = trainloader.dataset.tensors
    x_test = x_test[:,0]
    y_test = y_test[:,0]
    x_train = x_train[:,0]
    y_train = y_train[:,0]
    samples = []
    for network in nets:
        preds = network.forward(torch.tensor(x_test).float().cuda()).cpu().data.numpy()
        samples.append(preds)
        
    samples = np.array(samples)
    means = (samples.mean(axis = 0)).reshape(-1)
    var = (samples.std(axis = 0)).reshape(-1)

    plt.figure(figsize=(10,4))
    plt.plot(x_test, y_test, label='true function', color='black', alpha=0.7) #2ca02c
    plt.fill_between(x_test, means-var, means+var, alpha=0.3, label='$\pm 2 \sigma$', color='#ff7f0e')
    plt.plot(x_train, y_train, '*', color='#1f77b4', label='train data', alpha=0.6)
    plt.plot(x_test, means, label='mean', color='#ff7f0e')
    plt.title("SGLD")
    #plt.axvline(x=x_new[72], color='gray', alpha=0.7, linestyle='--')
    #plt.axvline(x=x_new[90], color='gray', alpha=0.7, linestyle='--')
    #plt.axvline(x=x_new[143], color='gray', alpha=0.7, linestyle='--')
    #plt.axvline(x=x_new[158], color='gray', alpha=0.7, linestyle='--')
    plt.legend(loc='lower left')
    plt.savefig("Regresion_SGLD.pdf", bbox_inches='tight')



def plot_Uncertainty_drop(x, target, test_iter, net, err_type_):
    degress = np.linspace(0,150, 20)
    noise_list = np.linspace(0,2, 20)
    if(err_type_=="noise"):
        input_list = noise_list
    elif(err_type_=="rotate"):
        input_list = degress
    H_mean_list = []
    H_var_list = []
    acc_list = []
    variation_ratio_list = []
    x_new_list = []
    softmax_means = []
    softmax_vars = []
    for noise in tqdm(input_list):
        if(err_type_=="noise"):
            x_new = torch.rand_like(x)*noise + x
        elif(err_type_=="rotate"):
            x_new = torchvision.transforms.functional.rotate(x, noise)
        x_new_list.append(x_new)
        _, H_mean, H_var, pred_list, acc, variation_ratio = test_MC(x_new, target, test_iter, net)
        H_mean_list.append(H_mean)
        H_var_list.append(H_var)
        acc_list.append(acc)
        variation_ratio_list.append(variation_ratio)
        softmax_means.append(torch.mean(pred_list[:,0,target], dim=0).cpu().data.numpy())
        softmax_vars.append(torch.std(pred_list[:,0,target], dim=0).cpu().data.numpy())

    H_mean_list = np.array(H_mean_list)
    H_var_list = np.array(H_var_list)
    softmax_means = np.array(softmax_means)
    softmax_vars = np.array(softmax_vars)
    
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(7)
    ax.set_title("MC DropOut", fontweight="bold")
    if(err_type_ == "noise"):
        ax.set_xlabel("noise")
    elif(err_type_ == "rotate"):
        ax.set_xlabel("rotation degree")
    ax.set_ylabel("nat", color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    lns1 = ax.plot(input_list, H_mean_list, color='#1f77b4', label="Entropy")
    ax.fill_between(input_list, H_mean_list[:,0]-H_var_list[:,0], H_mean_list[:,0]+H_var_list[:,0], alpha=0.3, color='#1f77b4')

    ax2 = ax.twinx()
    ax2.set_ylabel('predictive accuracy', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    lns2 = ax2.plot(input_list, softmax_means, label="softmaxt", color='#ff7f0e')
    ax2.fill_between(input_list, softmax_means-softmax_vars, softmax_means+softmax_vars, alpha=0.3, color='#ff7f0e')
    ax2.grid(None)

    ax.tick_params(axis='x', which='major', pad=26)
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    for i, c in enumerate(input_list):
        if(i%(len(input_list)//5) == 0 or i==len(input_list)-1):
          offset_image(c, c, ax, x_new_list[i])

    if(err_type_=="noise"):
        plt.savefig("Entropy_noise_DropOut.pdf", bbox_inches='tight')
    elif(err_type_=="rotate"):
        plt.savefig("Entropy_rotate_DropOut.pdf", bbox_inches='tight')


def plot_Uncertainty_Regression_drop(valloader, trainloader, net):
    x_test, y_test = valloader.dataset.tensors
    x_train, y_train = trainloader.dataset.tensors
    x_test = x_test[:,0]
    y_test = y_test[:,0]
    x_train = x_train[:,0]
    y_train = y_train[:,0]
    samples = []
    noises = []
    for i in range(100):
        preds = net.network.forward(torch.tensor(x_test).float().cuda()).cpu().data.numpy()
        samples.append(preds)
        
    samples = np.array(samples)
    means = (samples.mean(axis = 0)).reshape(-1)
    var = (samples.std(axis = 0)).reshape(-1)

    plt.figure(figsize=(10,4))
    plt.plot(x_test, y_test, label='true function', color='black', alpha=0.7) #2ca02c
    plt.fill_between(x_test, means-var, means+var, alpha=0.3, label='$\pm 2 \sigma$', color='#1f77b4')
    plt.plot(x_train, y_train, '*', color='#ff7f0e', label='train data', alpha=0.4)
    plt.plot(x_test, means, label='mean', color='#1f77b4') #1f77b4
    plt.title("MC Dropout")
    #plt.axvline(x=x_new[72], color='gray', alpha=0.7, linestyle='--')
    #plt.axvline(x=x_new[90], color='gray', alpha=0.7, linestyle='--')
    #plt.axvline(x=x_new[143], color='gray', alpha=0.7, linestyle='--')
    #plt.axvline(x=x_new[158], color='gray', alpha=0.7, linestyle='--')
    plt.legend(loc='lower left')
    plt.savefig("Regresion_Drop.pdf", bbox_inches='tight')
    

