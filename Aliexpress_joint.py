import torch
import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np
import argparse
from datetime import datetime
from datasets.AliexpressDataset import AliExpressDataset
from models.encoder.AliEncoder import AliEncoder
from models.decoder.AliDecoder import AliDecoder
import torch.nn.functional as F
from metrics.AUCMetric import AUCMetric
from collections import defaultdict
from utils.timer import TimeRecorder
from auxilearn.hypernet import MonoJoint
from auxilearn.optim import MetaOptimizer
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
import random
from utils.superloss import Superloss

random.seed(8876)

class SaveBestModel(object):
    def __init__(self, save_path):
        self.best_accuracy_improve = 0
        self.save_path = save_path
        self.best_epoch = 0
        self.ctr_base = 0.0
        self.ctcvr_base = 0.0        
        self.ctr_best = 0.0
        self.ctcvr_best = 0.0


    def savebestmodel(self, model, score1 ,score2, cur_epoch):
        if cur_epoch == 0:
            self.ctr_base = score1
            self.ctcvr_base = score2
            self.ctr_best = score1
            self.ctcvr_best = score2
            self.best_accuracy_improve = 0.0
            self.best_epoch = 0
            state={}
            for m in model:
                state[m] = model[m].state_dict()
            torch.save(state, self.save_path)
            return self.best_epoch
        
        impro1 = (score1-self.ctr_base)/self.ctr_base
        impro2 = (score2-self.ctcvr_base)/self.ctcvr_base
        cur_improve = (impro1+impro2)/2

        if cur_improve > self.best_accuracy_improve:
            self.best_epoch = cur_epoch
            self.best_accuracy_improve = cur_improve
            self.ctr_best = score1
            self.ctcvr_best = score2
            state={}
            for m in model:
                state[m] = model[m].state_dict()
            torch.save(state, self.save_path)
        return self.best_epoch

def grad2vec(origin_grad):
    return torch.cat([grad.flatten() for grad in origin_grad if grad is not None])

def cos_sim(grad1,grad2):
    if grad1.size(0) != grad2.size(0):
        size = max(grad1.size(0), grad2.size(0))
        gap = abs(grad1.size(0) - grad2.size(0))
        if grad1.size(0) == size:
            grad2 = torch.cat([grad2, torch.zeros(gap).to(grad2.device)])
        elif grad2.size(0) == size:
            grad1 = torch.cat([grad1, torch.zeros(gap).to(grad1.device)])
        grad1 = grad1.view(size, -1)
        grad2 = grad2.view(size, -1)
    return (F.cosine_similarity(grad1, grad2, dim=0)).squeeze()

def magnitude_sim(grad1,grad2):
    grad1_mag = torch.norm(grad1)
    grad2_mag = torch.norm(grad2)
    tmp1 = 2*grad1_mag*grad2_mag
    tmp2 = torch.square(grad1_mag)+torch.square(grad2_mag)
    msim=2*((tmp1/tmp2)-0.5)
    # msim=tmp1/tmp2
    return msim
  

def train(model, optimizer,aux_model, meta_optimizer, hyperstep,scheduler,cur_epoch,epoch,data_loader, device, log_interval=100):
    # model.train()
    for m in model:
        model[m].train()
    aux_model.train()
    # n_meta_train_loss_accum = 5 # accumulated training batch number for meta test
    total_loss = 0
    train_loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    batch_num = len(train_loader)
    hyperstep = batch_num // 15
    CTRsuperloss=Superloss()
    CTCVRsuperloss=Superloss()
    
    for cur_batch, (alidata, labels) in enumerate(train_loader):
        # save aux_img and aux_img_labels
        if (cur_batch!=0) & (cur_batch%hyperstep==0):
            aux_alidata = alidata.to(device)
            aux_alidata_labels=labels
            continue

        task_num = len(labels)
        alidata = alidata.to(device)
        encoder_output = model[0](alidata)
        pred = [ [] for _ in range(task_num) ]
        for i in range(task_num):
            pred[i] = model[i+1](encoder_output)
        ctrloss_list = F.binary_cross_entropy(pred[0],labels['CTR'].float().to(device),reduction='none')
        ctcvrloss_list = F.binary_cross_entropy(pred[1],labels['CTCVR'].float().to(device),reduction='none')
        
        superloss_list = [CTRsuperloss(ctrloss_list), CTCVRsuperloss(ctcvrloss_list)]

        mean_loss=[F.binary_cross_entropy(pred[0],labels['CTR'].float().to(device)), F.binary_cross_entropy(pred[1],labels['CTCVR'].float().to(device))]
        # mean_loss=[ctrloss_list.mean(),ctcvrloss_list.mean()]
       
        en_model_params = list(model[0].parameters())
        main_model_params = []
        for m in model:
            main_model_params += model[m].parameters()
        
        grad_list=[[] for _ in range(task_num)]
        cossim_list=[[] for _ in range(task_num)]
        magsim_list=[[] for _ in range(task_num)]
        for t in range(task_num):
            grad_list[t] = grad2vec(torch.autograd.grad(mean_loss[t], en_model_params, allow_unused=True,retain_graph=True))
        # caculate similarity
        for task_idx1 in range(task_num):
            for task_idx2 in range(task_idx1+1,task_num):
                cossim = cos_sim(grad_list[task_idx1],grad_list[task_idx2])
                magsim = magnitude_sim(grad_list[task_idx1],grad_list[task_idx2])
                cossim_list[task_idx1].append(cossim)
                cossim_list[task_idx2].append(cossim)
                magsim_list[task_idx1].append(magsim)
                magsim_list[task_idx2].append(magsim)

        cossim_list = [torch.stack(row) for row in cossim_list]
        magsim_list = [torch.stack(row) for row in magsim_list]
        cossim_list = [torch.sum(row)/(task_num-1) for row in cossim_list]
        magsim_list = [torch.sum(row)/(task_num-1) for row in magsim_list]
        final_loss = aux_model(cossim_list, magsim_list, superloss_list, to_train= True)
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        if (cur_batch!=0) & (cur_batch%hyperstep==0):
            meta_val_loss = 0.0
            encoder_output = model[0](aux_alidata)
            pred = [ [] for _ in range(task_num) ]
            for i in range(task_num):
                pred[i] = model[i+1](encoder_output)
            
            ctrloss_list = F.binary_cross_entropy(pred[0],aux_alidata_labels['CTR'].float().to(device),reduction='none')
            ctcvrloss_list = F.binary_cross_entropy(pred[1],aux_alidata_labels['CTCVR'].float().to(device),reduction='none')
            superloss_list = [CTRsuperloss(ctrloss_list), CTCVRsuperloss(ctcvrloss_list)]
            meta_val_loss = sum(superloss_list)
                        
            inner_meta_train_loss = 0.
            n_meta_train_loss_accum = 5 # default 1
            n_train_step = 0
            for data in data_loader:
                if n_train_step < n_meta_train_loss_accum:
                    n_train_step += 1
                    # print("start here")
                    alidata, labels = data
                    alidata = alidata.to(device)
                    encoder_output = model[0](alidata)
                    pred = [ [] for _ in range(task_num) ]
                    for i in range(task_num):
                        pred[i] = model[i+1](encoder_output)
                    
                    ctrloss_list = F.binary_cross_entropy(pred[0],aux_alidata_labels['CTR'].float().to(device),reduction='none')
                    ctcvrloss_list = F.binary_cross_entropy(pred[1],aux_alidata_labels['CTCVR'].float().to(device),reduction='none')
                    
                    inner_super_loss = [CTRsuperloss(ctrloss_list), CTCVRsuperloss(ctcvrloss_list)]
                    
                    mean_loss=[F.binary_cross_entropy(pred[0],labels['CTR'].float().to(device)), F.binary_cross_entropy(pred[1],labels['CTCVR'].float().to(device))]
                    # mean_loss=[ctrloss_list.mean(),ctcvrloss_list.mean()]
                    grad_list=[[] for _ in range(task_num)]
                    cossim_list=[[] for _ in range(task_num)]
                    magsim_list=[[] for _ in range(task_num)]
                    for t in range(task_num):
                        grad_list[t] = grad2vec(torch.autograd.grad(mean_loss[t], en_model_params, allow_unused=True,retain_graph=True))
                    # caculate similarity
                    for task_idx1 in range(task_num):
                        for task_idx2 in range(task_idx1+1,task_num):
                            cossim = cos_sim(grad_list[task_idx1],grad_list[task_idx2])
                            magsim = magnitude_sim(grad_list[task_idx1],grad_list[task_idx2])
                            cossim_list[task_idx1].append(cossim)
                            cossim_list[task_idx2].append(cossim)
                            magsim_list[task_idx1].append(magsim)
                            magsim_list[task_idx2].append(magsim)

                    cossim_list = [torch.stack(row) for row in cossim_list]
                    magsim_list = [torch.stack(row) for row in magsim_list]
                    cossim_list = [torch.sum(row)/(task_num-1) for row in cossim_list]
                    magsim_list = [torch.sum(row)/(task_num-1) for row in magsim_list]

                    meta_train_loss = aux_model(cossim_list, magsim_list, inner_super_loss, to_train= False)
                    inner_meta_train_loss += meta_train_loss
                else:
                    break
            meta_optimizer.step(
                    val_loss=meta_val_loss,
                    train_loss=inner_meta_train_loss,
                    aux_params = list(aux_model.parameters()),
                    parameters = main_model_params
                )
            aux_model.clamp()
        total_loss += final_loss.item()
        if (cur_batch + 1) % log_interval == 0:
            train_loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0
    scheduler.step()
    
def test(model, data_loader, task_num, device):
    # model.eval()
    for m in model:
        model[m].eval()
    CTRMetric = AUCMetric()
    CTCVRMetric = AUCMetric()
    target = {0:'CTR',1:'CTCVR'}
    testloader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    test_loss = [[] for _ in range(task_num)]
    all_test_preds, all_test_gts = defaultdict(list), defaultdict(list)
    concatenated_test_preds, concatenated_test_gts = defaultdict(), defaultdict()
    ctrauc = 0
    ctcvrauc = 0
    with torch.no_grad():
        for cur_batch, (alidata, alilabels) in enumerate(testloader):
            alidata = alidata.to(device)
            encoder_output = model[0](alidata)
            pred = [ [] for _ in range(task_num) ]
            for i in range(task_num):
                pred[i] = model[i+1](encoder_output)
                all_test_preds[i].append(pred[i])
                all_test_gts[i].append(alilabels[target[i]].float().to(device))
    for i in range(task_num):
        concatenated_test_preds[i] = torch.cat(all_test_preds[i], dim=0)
        concatenated_test_gts[i] = torch.cat(all_test_gts[i], dim=0)
    CTRMetric.update_fun(concatenated_test_preds[0],concatenated_test_gts[0])
    CTCVRMetric.update_fun(concatenated_test_preds[1],concatenated_test_gts[1])
    ctrauc = CTRMetric.score_fun()
    ctcvrauc = CTCVRMetric.score_fun()
    return ctrauc, ctcvrauc

def main(data_path, domain, task_num, epoch, learning_rate, 
         train_batch_size, test_batch_size, weight_decay, device, save_dir,aux_lr,auxw_decay):

    device = torch.device(device)
    # ------
    data_path = data_path+'/'+domain+'/'
    train_dataset = torch.load(data_path+'/cached_processed_dataset_train')
    test_dataset = torch.load(data_path+'/cached_processed_dataset_test')
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=16, shuffle=False)
    encoder = AliEncoder().to(device)
    decoder1 = AliDecoder().to(device)
    decoder2 = AliDecoder().to(device)
    model={}
    model[0] = encoder
    model[1] = decoder1
    model[2] = decoder2
    model_params = []
    for m in model:
        model_params += model[m].parameters()
    task_num = len(model)-1
    aux_model = MonoJoint(main_task=0, input_dim = task_num, weight_normalization=False, init_lower= 0.0, init_upper=1.0,device=device)

    optimizer = torch.optim.Adam(params=model_params, lr=learning_rate, weight_decay=weight_decay)
    optimizer_aux = torch.optim.Adam(aux_model.parameters(), lr=aux_lr, weight_decay=auxw_decay)
    meta_optimizer = MetaOptimizer(
    meta_optimizer= optimizer_aux,
    hpo_lr = learning_rate,
    truncate_iter = 3,  # default 3
    max_grad_norm = 10  # default 10
    )

    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    now=datetime.now()
    timestamp = datetime.timestamp(now)
    current_time = datetime.fromtimestamp(timestamp)
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    save_dir = save_dir+'_'+domain
    os.makedirs(f'{save_dir}/{time_str}')
    dataset_name = 'AliExpress_'+domain

    save_path=f'{save_dir}/{time_str}/{dataset_name}.pt'
    best_model = SaveBestModel(save_path=save_path)
    best_epoch=0
    log_path=f'{save_dir}/{time_str}/{dataset_name}.txt'
    with open(log_path, 'w') as f:  
        f.write(time_str+' '+str(learning_rate)+"\n") 
    
    f = open(log_path, 'a', encoding = 'utf-8')
    hyperstep = 80 
    aux_model.clamp()
    for epoch_i in range(epoch):
        s = 'epoch: {}/{}'.format(epoch_i,epoch-1)
        print(s)
        train(model, optimizer,aux_model, meta_optimizer, hyperstep,scheduler,epoch_i,epoch,train_data_loader, device)
        ctrauc, ctcvrauc = test(model, test_data_loader, task_num, device)
        print('lr:',optimizer.param_groups[0]['lr'])
        s = 'epoch: {}, CTR AUC: {}, CTCVR AUC: {} \n'.format(epoch_i, ctrauc, ctcvrauc)
        print(s) 
        f.write(s)
        best_epoch = best_model.savebestmodel(model,ctrauc,ctcvrauc,cur_epoch=epoch_i)
        print('current best epoch:',best_epoch)
        f.write('current best epoch:{}\n'.format(best_epoch))


    # model.load_state_dict(torch.load(save_path))
    # seg_score, dep_score , nor_score, loss = test(model, test_data_loader, task_num, device)
    ctr, ctcvr =best_model.ctr_best, best_model.ctcvr_best
    # f = open('{}_{}.txt'.format(model_name, dataset_name), 'a', encoding = 'utf-8')
    f.write('\n')
    f.write('time:{}\n'.format(time_str))
    f.write('learning rate: {}\n'.format(learning_rate))
    print('best epoch {}'.format(best_epoch))
    print('best ', 'ctr auc:', ctr, 'ctcvr auc',ctcvr)
    # for i in range(task_num):
    #     print('task {},  Log-loss {}'.format(i, loss[i]))
    #     f.write('task {}, Log-loss {}\n'.format(i, loss[i]))
    
    f.write('best epoch {}/{}, CTR AUC: {}, CTCVR AUC: {}\n'.format(best_epoch, epoch,ctr,ctcvr))
    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/admin/LiuZeYu/Datasets/AliExpress/data')
    parser.add_argument('--domain', default='ES',help='ES,FR,NL,US')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1.0e-4)
    parser.add_argument('--weight_decay', type=float, default=1.0e-5)


    # parser.add_argument('--n_meta_loss_accum', type = int, default = 1, help="accumulated batch number for meta test")
    parser.add_argument('--aux_lr', type= float , default= 1.0e-3, help='aux learning rate')
    parser.add_argument('--auxw_decay', type= float , default= 1.0e-5, help='aux weight decay')
    parser.add_argument('--hyperstep', type= int , default= 300, help='step num for aux model')

    # ---------------------------------------------------------
    parser.add_argument('--train_batch_size', type=int, default=4096) # 2048
    parser.add_argument('--test_batch_size', type=int, default=4096) # 2048
    # ---------------------------------------------------------
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='./chkpt/AliExpress')
    args = parser.parse_args()
    main(args.data_path,
         args.domain,
         args.task_num,
         args.epoch,
         args.learning_rate,
         args.train_batch_size,
         args.test_batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.aux_lr,
         args.auxw_decay,
         )