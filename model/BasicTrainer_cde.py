import torch
import math
import os
import time
import copy
import numpy as np
from metrics import metric
import lib.load_dataset
from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.TrainInits import print_model_parameters
import pandas as pd
import matplotlib.pyplot as plt
from lib.load_dataset import StandardScaler
from pynvml import *
from scipy.special import logit
def calc_da(actual, predicted, h=1):
    da_score = np.sign(actual[:, h:,:] - actual[:, :-h,:]) == np.sign(predicted[:, h:,:] - predicted[:, :-h,:])
    return np.mean(da_score)
def nvidia_info():
    # pip install nvidia-ml-py
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        nvmlInit()
        nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict

def check_gpu_mem_usedRate(init_gpu,init_gpu_per):
    max_rate = 0.0
    info = nvidia_info()
    used = info['gpus'][0]['used']
    tot = info['gpus'][0]['total']
    print(f"GPU0 used: {used}, tot: {tot}, Usage rate:{used/tot}")
    if used/tot > max_rate:
        max_rate = used/tot
    GPU_1 = used - init_gpu
    GPU_2 = max_rate - init_gpu_per
    print("Maximum usage of GPU0:", GPU_1)
    print("Maximum GPU0 utilization rate:", GPU_2)

def inverse_transform_1(mean,std,data):
    mean = torch.from_numpy(np.array(mean)).type_as(data).to(data.device) if torch.is_tensor(data) else mean
    std = torch.from_numpy(np.array(std)).type_as(data).to(data.device) if torch.is_tensor(data) else std
    return (data * std) + mean

class Trainer(object):
    def __init__(self, model,  loss, optimizer, train_loader, val_loader, test_loader,
                  args, lr_scheduler, device, times,
                 w,init_len,scaler,init_gpu,init_gpu_per):
        super(Trainer, self).__init__()
        self.model = model

        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        total_param = print_model_parameters(model, only_num=False)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)
        self.logger.info(self.model)
        self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        self.times = times.to(self.device, dtype=torch.float)
        self.w = w
        self.init_gpu = init_gpu
        self.init_gpu_per = init_gpu_per

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        LOSS_LIST = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
                *valid_coeffs, val_init , target = batch
                label = torch.squeeze(target,-1)
                output,ww ,lambd,fract= self.model(self.times, valid_coeffs, val_init)
                output = torch.squeeze(output, -1)
                output = output.reshape(output.size(0) * output.size(1), output.size(2), output.size(3))
                label = label.transpose(0,1)
                loss = self.loss(output.cuda(), label)
                LOSS_LIST.append(loss)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {}'.format(epoch, val_loss))
        if self.args.tensorboard:
            self.w.add_scalar(f'valid/loss', val_loss, epoch)
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
            *train_coeffs, train_init, target = batch
            label = torch.squeeze(target,-1)
            mask = torch.any(label != 0, dim=2)
            self.optimizer.zero_grad()

            output,ww,lambd,fract = self.model(self.times, train_coeffs, train_init)
            output = torch.squeeze(output,-1)
            output = output.reshape(output.size(0)*output.size(1),output.size(2),output.size(3))
            label = label.transpose(0,1)



            loss = self.loss(output[...,:].cuda(), label[...,:])
            loss.backward()

            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()


            if batch_idx % self.args.log_step == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        check_gpu_mem_usedRate(self.init_gpu,self.init_gpu_per)

        train_epoch_loss = total_loss/self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {}'.format(epoch, train_epoch_loss))
        if self.args.tensorboard:
            self.w.add_scalar(f'train/loss', train_epoch_loss, epoch)

        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            train_epoch_loss = self.train_epoch(epoch)
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            if best_state == True:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
            


        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))


        if not self.args.debug:
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
        self.model.load_state_dict(best_model)
        self.test(self.model, self.args, self.test_loader, self.scaler, self.logger, None, self.times)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }


    @staticmethod
    def test(model, args, data_loader, scaler, logger, path, times):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                *test_coeffs,test_init, target = batch
                label = torch.squeeze(target,-1)
                output,ww ,lambd,fract= model(times.to(args.device, dtype=torch.float), test_coeffs,test_init)
                output = torch.squeeze(output,-1)
                output = output.reshape(output.size(0)*output.size(1),output.size(2),output.size(3))
                output = output.transpose(0, 1)

                y_true.append(label)
                y_pred.append(output)


        y_pred = torch.cat(y_pred,dim=0)
        y_true = torch.cat(y_true,dim=0)
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred_result = y_pred.reshape(y_pred.shape[0],y_pred.shape[1],y_pred.shape[2])
        y_true_result = y_true.reshape(y_true.shape[0],y_true.shape[1],y_true.shape[2])
        mae, mse, rmse, mape, mspe = metric(y_pred_result[...,:], y_true_result[...,:])
        print('mse:{}, mae:{},rmse:{}'.format(mse, mae,rmse))

        overall_da = calc_da(y_pred_result[..., :],y_true_result[..., :], h=1)
        print("DA:", overall_da)



