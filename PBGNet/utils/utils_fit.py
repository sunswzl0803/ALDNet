import time
from datetime import datetime
import numpy as np
import torch
from cv2 import imread
import numpy as np
import os
import torch.nn.functional as F
import ast
import torch.utils.data as data
from torch import nn
from nets.pbgnet_training import CE_Loss, Dice_loss, Focal_Loss, BCEDice_Loss, structure_loss
from tqdm import tqdm
from utils.utils_metrics import f_score
from utils.utils import get_lr

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)
    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):]))

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, local_rank=0, num_train=None, num_val=None):
    total_loss = 0
    total_f_score = 0
    end = time.time()
    time_sum = 0
    val_loss = 0
    val_f_score = 0
    batch_size=4
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    loss_record_final, loss_record1, loss_record2, loss_record3, loss_record4,loss_record_ppd = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for iteration, batch in enumerate(gen,start=1):
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        optimizer.zero_grad()
        if fp16:
            from torch.cuda.amp import autocast
            with autocast():
                # ----------------------#
                #   前向传播
                # ----------------------#
                final_map, lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4, lateral_map_ppd = model(imgs)
                # ----------------------#
                #   损失计算
                # ----------------------#
                if focal_loss:
                     pass
                if dice_loss:
                     pass
                # ---------------------------------------------------------------------
                loss_ppd = structure_loss(lateral_map_ppd, labels)
                loss_4 = structure_loss(lateral_map_4, labels)
                loss_3 = structure_loss(lateral_map_3, labels)
                loss_2 = structure_loss(lateral_map_2, labels)
                loss_1 = structure_loss(lateral_map_1, labels)
                loss_final = structure_loss(final_map, labels)
                loss = (loss_final + loss_1 + loss_2 + loss_3 + loss_4 + loss_ppd) / 6# TODO: try different weights for loss
                # ---------------------------------------------------------------------------
                # with torch.no_grad():
                #     # -------------------------------#
                #     #   计算f_score
                #     # -------------------------------#
                #     _f_score = f_score(outputs, labels)
            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            clip_gradient(optimizer, 0.5)
            scaler.update()
        total_loss += loss.item()
        # total_f_score += _f_score.item()
        loss_record_final.update(loss_final.data, 4)
        loss_record1.update(loss_1.data, 4)
        loss_record2.update(loss_2.data, 4)
        loss_record3.update(loss_3.data, 4)
        loss_record4.update(loss_4.data, 4)
        loss_record_ppd.update(loss_ppd.data, 4)
        if local_rank == 0:
            if iteration % 1 == 0 or iteration == epoch_step:
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[loss_final: {:.4f}, loss_1: {:0.4f}, loss_2: {:0.4f}, loss_3: {:0.4f}, loss_4: {:0.4f}, loss_ppd: {:0.4f}]'.
                      format(epoch+1, 100, iteration, epoch_step,
                             loss_record_final.show(), loss_record1.show(),loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record_ppd.show()))
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val,start=1):
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            # ----------------------#
            #   前向传播
            # ----------------------#
            final_map, lateral_map_1, lateral_map_2, lateral_map_3, lateral_map_4, lateral_map_ppd = model(imgs)
            # ----------------------#
            #   损失计算
            # ----------------------#
            if focal_loss:
                pass
            if dice_loss:
                pass
            loss_ppd = structure_loss(lateral_map_ppd, labels)
            loss_4 = structure_loss(lateral_map_4, labels)
            loss_3 = structure_loss(lateral_map_3, labels)
            loss_2 = structure_loss(lateral_map_2, labels)
            loss_1 = structure_loss(lateral_map_1, labels)
            loss_final = structure_loss(final_map, labels)
            loss = (loss_final + loss_1 + loss_2 + loss_3 + loss_4 + loss_ppd) / 6# TODO: try different weights for loss

            #   计算f_score
            # -------------------------------#
            # _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            # val_f_score += _f_score.item()
            loss_record_final.update(loss_final.data, 4)
            loss_record1.update(loss_1.data, 4)
            loss_record2.update(loss_2.data, 4)
            loss_record3.update(loss_3.data, 4)
            loss_record4.update(loss_4.data, 4)
            loss_record_ppd.update(loss_ppd.data, 4)
        if local_rank == 0:
            if iteration % 1 == 0 or iteration == epoch_step_val:
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                      '[loss_final: {:.4f}, loss_1: {:0.4f}, loss_2: {:0.4f}, loss_3: {:0.4f}, loss_4: {:0.4f}, loss_ppd: {:0.4f}]'.
                      format(epoch+1, 100, iteration, epoch_step_val,
                             loss_record_final.show(), loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record_ppd.show()))
            # pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
            #                     'f_score': val_f_score / (iteration + 1),
            #                     'lr': get_lr(optimizer)})
            # pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        # 求平均
        # print('Total Dice: %.4f || Val Dice: %.4f ' % (total_dice_score / epoch_step, val_dice_score / epoch_step_val))
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.4f-val_loss%.4f.pth' % (
            (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
