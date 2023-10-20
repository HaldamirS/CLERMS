import torch
import numpy as np
from utils.logger import Logger
from tqdm.notebook import tqdm
from tools.SupContrast import *


class EarlyStop:
    def __init__(self, tol_num, min_is_best):
        self.tol_num = tol_num
        self.best = np.finfo(np.float).max if min_is_best else np.finfo(np.float).min
        self.count = 0
        self.min_is_best = min_is_best

    def reach_stop_criteria(self, loss):
        if (self.min_is_best and loss > self.best) or (
            not self.min_is_best and loss < self.best
        ):
            self.count += 1
        else:
            self.best = loss
        return True if self.count >= 5 else False


class TrainerCon:
    def __init__(self, model, dataloader,score_df, optimizer, max_epoch, scheduler, dtype,pth,model_name = "best.pkl",sim_ratio=0, temperature=0.05):
        self.dtype = dtype
        self.model = model
        self.dataloader_train, self.dataloader_val = dataloader
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.scheduler = scheduler
        self.device = next(model.parameters()).device
        self.sup_con_loss = SupConLoss(contrast_mode='one',temperature=temperature, base_temperature=temperature)
        self.sim_loss = nn.MSELoss(reduction='mean')
        self.earlystop = EarlyStop(5, True)
        self.dict_pth = pth
        self.model_save_path = self.dict_pth + model_name
        self.logger = Logger().get_logger("ms2deepscore", self.dict_pth + "log.txt")
        self.score_df = score_df.values
        self.ratio = sim_ratio


    @torch.no_grad()
    def evaluate(self):
        self.model.train()
        losses = []
        for data in tqdm(self.dataloader_val):
            keys = data[-1]
            scores = torch.tensor(self.score_df[keys][:,keys]).to('cuda').to(torch.float32)
            data = data[:-1]
            self.optimizer.zero_grad()
            res = []
            for i in data:
                res.append(self.model(i).unsqueeze(1))
            res = torch.cat(res, dim=1)
            all_res = F.normalize(res.reshape(-1,res.shape[-1]),dim=-1)
            score_res = torch.matmul(all_res, all_res.T)
            score_label = scores.repeat_interleave(2,0).repeat_interleave(2,1)
            #score_label = torch.tile(scores,(2,2))
            #print(self.sup_con_loss(res, scores))
            #print(self.sim_loss(score_res, score_label))
            loss = self.sup_con_loss(res, scores) + self.ratio*self.sim_loss(score_res, score_label)

            # loss = self.sup_con_loss(res, scores)
            losses.append(loss.to('cpu').item())
        return np.mean(losses)


    def train_and_evaluate(self):
        f_str = " Epoch {:02}/{:02} Train Loss: {:.6f} ConLoss: {:.6f} SimLoss: {:.6f}"
        f_str_v = "Valid loss: {:.6f}"
        min_loss_valid =100
        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            losses = []
            conlosses = []
            simlosses = []
            for data in tqdm(self.dataloader_train):
                keys = data[-1]
                scores = torch.tensor(self.score_df[keys][:,keys]).to('cuda').to(torch.float32)
                #print(keys)
                #print(self.score_df.head())
                data = data[:-1]
                self.optimizer.zero_grad()
                res = []
                for i in data:
                    res.append(self.model(i).unsqueeze(1))
                res = torch.cat(res, dim=1)
                all_res = F.normalize(res.reshape(-1,res.shape[-1]),dim=-1)
                score_res = torch.matmul(all_res, all_res.T)
                score_label = scores.repeat_interleave(2,0).repeat_interleave(2,1)
                #score_label = torch.tile(scores,(2,2))
                conloss = self.sup_con_loss(res, scores)
                simloss = self.sim_loss(score_res, score_label)
                loss = conloss + self.ratio*simloss
                loss.backward()
                self.optimizer.step()
                losses.append(loss.to("cpu").item())
                conlosses.append(conloss.to("cpu").item())
                simlosses.append(simloss.to("cpu").item())
            self.logger.info(f_str.format(epoch, self.max_epoch, np.mean(losses), np.mean(conlosses), np.mean(simlosses)))
            valid = self.evaluate()
            if valid < min_loss_valid:
                min_loss_valid = valid
                torch.save(self.model.state_dict(), self.model_save_path + "/models/" + "best.pkl")             
            # torch.save(self.model.state_dict(), self.model_save_path + "/models/" + str(epoch)+".pkl")             

            self.logger.info(f_str_v.format(valid))

            if self.scheduler:
                self.scheduler.step()
