from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.fusion_daft import FusionDAFT
from models.ehr_models import LSTM
from models.cxr_models import CXRModels
from .trainer import Trainer


import numpy as np
from sklearn import metrics

class DAFTTrainer(Trainer):
    def __init__(self, 
        train_dl, 
        val_dl, 
        args,
        test_dl=None,
        train_iter=None,
        eval_iter=None,
        ):

        super(DAFTTrainer, self).__init__(args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_iter = eval_iter
        self.train_iter = train_iter
        self.args = args
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.ehr_model = LSTM(input_dim=76, num_classes=args.num_classes, hidden_dim=args.dim, dropout=args.dropout, layers=args.layers).to(self.device)
        self.cxr_model = CXRModels(self.args, self.device).to(self.device)



        self.model = FusionDAFT(args, self.ehr_model, self.cxr_model ).to(self.device)

        self.loss = nn.BCELoss()

        self.optimizer = optim.Adam(self.model.parameters(), args.lr, betas=(0.9, self.args.beta_1))
    
        self.load_state()
        print(self.optimizer)
        print(self.loss)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10, mode='min')

        self.best_auroc = 0
        self.best_stats = None

        if self.args.pretrained:
            self.load_ehr_pheno()
            self.load_cxr_pheno()
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99) 

        self.epochs_stats = {'loss train': [], 'loss val': [],  'auroc val': []}
   
    def step(self, optim, pred, y):
        loss = self.loss(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    def save_checkpoint(self, prefix='best'):
        path = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'best_auroc': self.best_auroc, 
            'optimizer' : self.optimizer.state_dict(),
            'epochs_stats': self.epochs_stats,
            # 'optimizer_ehr' : self.optimizer_ehr.state_dict(),
            }, path)
        print(f"saving {prefix} checkpoint at epoch {self.epoch}")

    def train_epoch(self):
        print(f'starting train epoch {self.epoch}')
        epoch_loss = 0
        steps = len(self.train_dl)
        for i, (x, img, y_ehr, y_cxr, seq_lengths, _) in enumerate (self.train_dl):
            y = self.get_gt(y_ehr, y_cxr)
            x = torch.from_numpy(x).float()
            x = x.to(self.device)
            y = y.to(self.device)
            
            img = img.to(self.device)

            output = self.model(x, seq_lengths, img)
            pred = output['daft_fusion'].squeeze()
            loss = self.step(self.optimizer, pred, y)

            

            epoch_loss = epoch_loss + loss.item()
           
            if self.train_iter is not None and (i+1) % self.train_iter == 0:
                # print(f'evaluation after {i} iteration')
                # self.eval_script()
                break
            if i % 100 == 9:
                eta = self.get_eta(self.epoch, i)
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] [{i:04}/{steps}] eta: {eta:<20}  lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/i:0.5f}")
        
        self.epochs_stats['loss train'].append(epoch_loss/i)
        
    
    def validate(self, dl, full_run=False):
        print(f'starting val epoch {self.epoch}')
        epoch_loss = 0
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for i, (x, img, y_ehr, y_cxr, seq_lengths, _) in enumerate (dl):
                y = self.get_gt(y_ehr, y_cxr)

                x = torch.from_numpy(x).float()
                x = Variable(x.to(self.device), requires_grad=False)
                
                y = Variable(y.to(self.device), requires_grad=False)
                img = img.to(self.device)

                output = self.model(x, seq_lengths, img)
                pred = output['daft_fusion'].squeeze()
                
                loss = self.loss(pred, y)
                
                epoch_loss += (loss.item() )#+ loss2.item() + loss3.item())/3)
                outPRED = torch.cat((outPRED, pred), 0)
                


                outGT = torch.cat((outGT, y), 0)
                
                if self.eval_iter is not None and (i+1) % self.eval_iter == 0 and not full_run:
                    break

        print(f"val [{self.epoch:04d} / {self.args.epochs:04d}] validation loss: \t{epoch_loss/i:0.5f}")
        ret = self.computeAUROC(outGT.data.cpu().numpy(), outPRED.data.cpu().numpy(), 'validation')
        
        self.epochs_stats['loss val'].append(epoch_loss/i)
        self.epochs_stats['auroc val'].append(ret['auroc_mean'])

        return ret

    def eval(self):
        print('validating ... ')
        self.epoch = 0
        self.model.eval()
        ret = self.validate(self.val_dl, full_run=True)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} val', filename='results_val.txt')
        self.model.eval()
        ret = self.validate(self.test_dl, full_run=True)
        self.print_and_write(ret , isbest=True, prefix=f'{self.args.fusion_type} test', filename='results_test.txt')
        return
    def train(self):

            # pred = output[self.args.fusion_type].squeeze()
        print(f'running for fusion_type {self.args.fusion_type}')
        for self.epoch in range(self.start_epoch, self.args.epochs):
            
            

            full_run = False if (self.args.task == 'decompensation' or self.args.task == 'length-of-stay') else True
            self.model.eval()
            ret = self.validate(self.val_dl, full_run=full_run)
            self.save_checkpoint(prefix='last')

            if self.best_auroc < ret['auroc_mean']:
                self.best_auroc = ret['auroc_mean']
                self.best_stats = ret
                self.save_checkpoint()
                # print(f'saving best AUROC {ret["ave_auc_micro"]:0.4f} checkpoint')
                self.print_and_write(ret, isbest=True)
                self.patience = 0

            else:
                self.print_and_write(ret, isbest=False)
                self.patience+=1

            self.model.train()
            self.train_epoch()
            
            self.plot_stats(key='loss', filename='loss.pdf')
            self.plot_stats(key='auroc', filename='auroc.pdf')
            if self.patience >= self.args.patience:
                break
        self.print_and_write(self.best_stats , isbest=True)

        
    

