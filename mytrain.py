# encoding: utf-8

"""
The main CheXNet model implementation.
"""
import re
import sys
import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from myread_data import DatasetGenerator_Classify
from sklearn.metrics import roc_auc_score
from skimage.measure import label
from mymodel import DenseNet121, Fusion_Branch
from PIL import Image
from tensorboardX import SummaryWriter
device = torch.device('cuda')

def calculate_metric(gtnp, pdnp):
    # input are numpy vector
    total_samples = len(gtnp)
    #print(f"Total sample: {total_samples}")
    total_correct = np.sum(gtnp == pdnp)
    accuracy = total_correct / total_samples
    gt_pos = np.where(gtnp == 1)[0]
    gt_neg = np.where(gtnp == 0)[0]
    TP = np.sum(pdnp[gt_pos])
    TN = np.sum(1 - pdnp[gt_neg])
    FP = np.sum(pdnp[gt_neg])
    FN = np.sum(1 - pdnp[gt_pos])
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    metrics = {}
    metrics['accuracy'] = str(accuracy)
    metrics['precision'] = str(precision)
    metrics['recall'] = str(recall)
    metrics['f1'] = str(f1)
    metrics['tp'] = str(int(TP))
    metrics['tn'] = str(int(TN))
    metrics['fp'] = str(int(FP))
    metrics['fn'] = str(int(FN))

    return metrics

#np.set_printoptions(threshold = np.nan)

CKPT_PATH_G = ''#'/best_model/AG_CNN_Global_epoch_1.pkl' 
CKPT_PATH_L = ''#'/best_model/AG_CNN_Local_epoch_2.pkl' 
CKPT_PATH_F = ''#'/best_model/AG_CNN_Fusion_epoch_23.pkl'
N_CLASSES = 1

BATCH_SIZE = 80
RESIZE_SIZE = 256
CROP_SIZE = 224
MAX_EPOCH = 60
DATA_DIR = '/mnt/sda/hong01-data/CXR/images'
TRAIN_TXT = '../Multitask-Learning-CXR/dataset/binary_train.txt'
VAL_TXT = '../Multitask-Learning-CXR/dataset/binary_validate.txt'
TEST_TXT = '../Multitask-Learning-CXR/dataset/binary_test.txt'
CHEXNET_CHECKPOINT = '../Multitask-Learning-CXR/pretrained_chexnet_model/m-25012018-123527.pth.tar'
CHECKPOINT = None #'/home/hong01/LungProject/Multitask-Learning-CXR/trained_model/model_0/classify_15062020-235605.pth.tar' #None # 'model/classify_06062020-015937.pth.tar' #'model/classify_05062020-064630.pth.tar'
SAVE_DIR = 'trained_model/'
SAVE_MODEL_NAME = 'AG_CNN'

timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime

if not os.path.exists(f'report/{SAVE_MODEL_NAME}'):
    os.makedirs(f'report/{SAVE_MODEL_NAME}')

if not os.path.exists(f'tensorboard/{SAVE_MODEL_NAME}'):
    os.makedirs(f'tensorboard/{SAVE_MODEL_NAME}')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

f_log = open(f"report/{SAVE_MODEL_NAME}/{SAVE_MODEL_NAME}-{timestampLaunch}-report.log", "w")
writer = SummaryWriter(f'tensorboard/{SAVE_MODEL_NAME}/{SAVE_MODEL_NAME}-{timestampLaunch}/')


LR_G = 1e-4
LR_L = 1e-4
LR_F = 1e-3

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize(RESIZE_SIZE),
   transforms.CenterCrop(CROP_SIZE),
   transforms.ToTensor(),
   normalize,
])


def Attention_gen_patchs(ori_image, fm_cuda):
    # fm => mask =>(+ ori-img) => crop = patchs
    feature_conv = fm_cuda.data.cpu().numpy()
    size_upsample = (224, 224) 
    bz, nc, h, w = feature_conv.shape

    patchs_cuda = torch.FloatTensor().cuda()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h,w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:,0])
        minw = min(ind[:,1])
        maxh = max(ind[:,0])
        maxw = max(ind[:,1])
        
        # to ori image 
        image = ori_image[i].numpy().reshape(224,224,3)
        image = image[int(224*0.334):int(224*0.667),int(224*0.334):int(224*0.667),:]

        image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh,minw:maxw,:] * 255 # because image was normalized before
        image_crop = preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB')) 

        img_variable = torch.autograd.Variable(image_crop.reshape(3,224,224).unsqueeze(0).cuda())

        patchs_cuda = torch.cat((patchs_cuda,img_variable),0)

    return patchs_cuda


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap , 0 , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 


def main():
    print('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.Resize(CROP_SIZE))
    transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)      
    transformSequence=transforms.Compose(transformList)

    transformList = []
    transformList.append(transforms.Resize(CROP_SIZE))
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequenceVal=transforms.Compose(transformList)
    
    train_dataset = DatasetGenerator_Classify(DATA_DIR, TRAIN_TXT, transformSequence)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = DatasetGenerator_Classify(DATA_DIR, VAL_TXT, transformSequenceVal)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=4, pin_memory=True)
    print('********************load data succeed!********************')


    print('********************load model********************')
    # initialize and load the model
    Global_Branch_model = DenseNet121(classCount=1, chexnet_pretrained=CHEXNET_CHECKPOINT).cuda()
    Local_Branch_model = DenseNet121(classCount=1, chexnet_pretrained=CHEXNET_CHECKPOINT).cuda()
    Fusion_Branch_model = Fusion_Branch(input_size = 2048, output_size = N_CLASSES).cuda()

    if os.path.isfile(CKPT_PATH_G):
        checkpoint = torch.load(CKPT_PATH_G)
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_L):
        checkpoint = torch.load(CKPT_PATH_L)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.isfile(CKPT_PATH_F):
        checkpoint = torch.load(CKPT_PATH_F)
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    #-------------------- SETTINGS: LOSS
    pos_weight = torch.FloatTensor([1.15]).to(device)
    #pos_weight = None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer_global = optim.Adam(Global_Branch_model.parameters(), lr=LR_G, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_global = lr_scheduler.StepLR(optimizer_global , step_size = 10, gamma = 0.5)
    
    optimizer_local = optim.Adam(Local_Branch_model.parameters(), lr=LR_L, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_local = lr_scheduler.StepLR(optimizer_local , step_size = 10, gamma = 0.5)
    
    optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(), lr=LR_F, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_fusion = lr_scheduler.StepLR(optimizer_fusion , step_size = 10, gamma = 0.2)
    print('********************load model succeed!********************')

    
    lossGlobalMin = 1000
    lossAllMin = 1000
    
    print('********************begin training!********************')
    for epoch in range(MAX_EPOCH):
        print('Epoch {}/{}'.format(epoch , MAX_EPOCH - 1))
        print('-' * 10)
        #set the mode of model
        lr_scheduler_global.step()  #about lr and gamma
        lr_scheduler_local.step() 
        lr_scheduler_fusion.step() 
        Global_Branch_model.train()  #set model to training mode
        Local_Branch_model.train()
        Fusion_Branch_model.train()

        lossAll = lossG = lossL = lossF = 0.0
        count = 0
        #Iterate over data
        for i, (input, target) in enumerate(train_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            optimizer_fusion.zero_grad()

            # compute output
            output_global, fm_global, pool_global = Global_Branch_model(input_var)
         
            patchs_var = Attention_gen_patchs(input,fm_global)

            output_local, _, pool_local = Local_Branch_model(patchs_var)
            #print(fusion_var.shape)
            output_fusion = Fusion_Branch_model(pool_global, pool_local)
            
            output_global = torch.sigmoid(output_global)
            output_local = torch.sigmoid(output_local)
            output_fusion = torch.sigmoid(output_fusion)

            # loss
            loss1 = criterion(output_global, target_var)
            loss2 = criterion(output_local, target_var)
            loss3 = criterion(output_fusion, target_var)
            #

            loss = loss1*0.8 + loss2*0.1 + loss3*0.1 

            if (i%500) == 0: 
                print('step: {} totalloss: {loss:.3f} loss1: {loss1:.3f} loss2: {loss2:.3f} loss3: {loss3:.3f}'.format(i, loss = loss, loss1 = loss1, loss2 = loss2, loss3 = loss3))

            loss.backward() 
            optimizer_global.step()  
            optimizer_local.step()
            optimizer_fusion.step()

            #print(loss.data.item())
            lossAll += loss.data.item()
            lossG += loss1.data.item()
            lossL += loss2.data.item()
            lossF += loss3.data.item()
            count += 1
            
            #break
            '''
            if i == 40:
                print('break')
                break
            '''
        lossAll /= count
        lossG /= count
        lossF /= count
        lossL /= count
    
        print('*******testing!*********')
        l1, l2, l3, l, a1, a2, a3 = test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model, test_loader, criterion)
        #break
        
        subinfo = ""
        if l1 < lossGlobalMin:
            lossGlobalMin = l1
            torch.save(Global_Branch_model.state_dict(), SAVE_DIR+'/'+SAVE_MODEL_NAME+'_Global_BEST.pth.tar')
            subinfo += "[Save Global BEST]"
        if l < lossAllMin:
            lossAllMin = l
            torch.save(Global_Branch_model.state_dict(), SAVE_DIR+'/'+SAVE_MODEL_NAME+'_Global.pth.tar')
            torch.save(Local_Branch_model.state_dict(), SAVE_DIR+'/'+SAVE_MODEL_NAME+'_Local.pth.tar')
            torch.save(Fusion_Branch_model.state_dict(), SAVE_DIR+'/'+SAVE_MODEL_NAME+'_Fusion.pth.tar')
            subinfo += "[Save All BEST]"
        
        info = f"[{epoch+1}]\nLoss Train All: {lossAll}\nLoss Train Global: {lossG}\nLoss Train Local: {lossL}\nLoss Train Fusion: {lossF}\n"
        print(info)
        info += f"Loss Val All: {l}\nLoss Val Global: {l1}\nLoss Val Local: {l2}\nLoss Val Fusion: {l3}\n"
        info += f"AUC Global: {a1}\nAUC Local: {a2}\nAUC Fusion: {a3}\n"
        info += subinfo
        info += "\n===================\n"
        
        f_log.write(info)
        
        writer.add_scalars('Loss Global', {'train': lossG}, epoch)
        writer.add_scalars('Loss Global', {'val': loss1}, epoch)
        writer.add_scalars('Loss Local', {'train': lossL}, epoch)
        writer.add_scalars('Loss Local', {'val': loss2}, epoch)
        writer.add_scalars('Loss Fusion', {'train': lossF}, epoch)
        writer.add_scalars('Loss Fusion', {'val': loss3}, epoch)
        writer.add_scalars('Loss All', {'train': lossAll}, epoch)
        writer.add_scalars('Loss All', {'val': loss}, epoch)
        writer.add_scalars('AUC Val', {'Global': a1}, epoch)
        writer.add_scalars('AUC Val', {'Local': a2}, epoch)
        writer.add_scalars('AUC Val', {'Fusion': a3}, epoch)
        
        current_lr_g = optimizer_global.param_groups[0]['lr']
        current_lr_l = optimizer_local.param_groups[0]['lr']
        current_lr_f = optimizer_fusion.param_groups[0]['lr']
        
        writer.add_scalars('Learning Rate', {'Global': current_lr_g}, epoch)
        writer.add_scalars('Learning Rate', {'Local': current_lr_l}, epoch)
        writer.add_scalars('Learning Rate', {'Fusion': current_lr_f}, epoch)
    

def test(model_global, model_local, model_fusion, test_loader, loss_func):

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor().cuda()
    pred_global = torch.FloatTensor().cuda()
    pred_local = torch.FloatTensor().cuda()
    pred_fusion = torch.FloatTensor().cuda()

    # switch to evaluate mode
    model_global.eval()
    model_local.eval()
    model_fusion.eval()
    cudnn.benchmark = True
    
    count = 0
    lossAll = lossG = lossL = lossF = 0

    for i, (inp, target) in enumerate(test_loader):
        with torch.no_grad():
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            input_var = torch.autograd.Variable(inp.cuda())
            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)
            
            patchs_var = Attention_gen_patchs(inp,fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global,pool_local)
            
            output_global = torch.sigmoid(output_global)
            output_local = torch.sigmoid(output_local)
            output_fusion = torch.sigmoid(output_fusion)
            
            loss1 = loss_func(output_global, target)
            loss2 = loss_func(output_local, target)
            loss3 = loss_func(output_fusion, target)
            
            loss = 0.8*loss1 + 0.1*loss2 + 0.1*loss3
            
            count += 1
            
            lossAll += loss.data.item()
            lossG += loss1.data.item()
            lossL += loss2.data.item()
            lossF += loss3.data.item()
            
            pred_global = torch.cat((pred_global, output_global.data), 0)
            pred_local = torch.cat((pred_local, output_local.data), 0)
            pred_fusion = torch.cat((pred_fusion, output_fusion.data), 0)
    
    outGTnp = gt.cpu().numpy()
    outPREDnp = pred_global.cpu().numpy()
    aurocIndividual = computeAUROC(outGTnp, outPREDnp, classCount=1)
    auroc_global = np.array(aurocIndividual).mean()
    
    outPREDnp = pred_local.cpu().numpy()
    aurocIndividual = computeAUROC(outGTnp, outPREDnp, classCount=1)
    auroc_local = np.array(aurocIndividual).mean()
    
    outPREDnp = pred_fusion.cpu().numpy()
    aurocIndividual = computeAUROC(outGTnp, outPREDnp, classCount=1)
    auroc_fusion = np.array(aurocIndividual).mean()
    
    lossAll /= count
    lossG /= count
    lossF /= count
    lossL /= count
    
    return lossG, lossL, lossF, lossAll, auroc_global, auroc_local, auroc_fusion


def computeAUROC (dataGT, dataPRED, classCount):
    
    outAUROC = []
    '''
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    '''
    for i in range(classCount):
        outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        
    return outAUROC

if __name__ == '__main__':
    main()