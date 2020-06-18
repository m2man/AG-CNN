import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import torchvision
from torchsummary import summary
# from initialize import *

device = torch.device('cuda')

class DenseNet121(nn.Module):
    def __init__(self, classCount, chexnet_pretrained=None, isTrained=True):
        super(DenseNet121, self).__init__()
        # Init model
        if chexnet_pretrained:
            isTrained=False
        densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        # Extract feature size
        kernelCount = densenet121.classifier.in_features
        # take specific modules
        self.features = densenet121.features
        self.classifier = nn.Linear(kernelCount, classCount)
        # lead pretrained chexnet
        if chexnet_pretrained:
            modelCheckpoint = torch.load(chexnet_pretrained) #torch.load('../pretrained_chexnet_model/m-25012018-123527.pth.tar')
            new_state_dict = OrderedDict()
            ##### Convert Parrallel to Single GPU Loading Model #####
            for k, v in modelCheckpoint['state_dict'].items():
                if 'features' in k:
                    if 'module.' in k:
                        name = k[28:] # remove `module.densenet121.features`
                        name = name.replace('norm.', 'norm')
                        name = name.replace('conv.', 'conv')
                        name = name.replace('normweight', 'norm.weight')
                        name = name.replace('convweight', 'conv.weight')
                        name = name.replace('normbias', 'norm.bias')
                        name = name.replace('normrunning_mean', 'norm.running_mean')
                        name = name.replace('normrunning_var', 'norm.running_var')
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
            self.features.load_state_dict(new_state_dict)
        # del variables
        del densenet121
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out_after_pooling = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1) # last layer have size of 7x7
        out = self.classifier(out_after_pooling)
        return out, features, out_after_pooling
    
class Fusion_Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super(Fusion_Branch, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, global_pool, local_pool):
        #fusion = torch.cat((global_pool.unsqueeze(2), local_pool.unsqueeze(2)), 2).cuda()
        #fusion = fusion.max(2)[0]#.squeeze(2).cuda()
        #print(fusion.shape)
        fusion = torch.cat((global_pool,local_pool), 1).cuda()
        fusion_var = torch.autograd.Variable(fusion)
        x = self.fc(fusion_var)

        return x