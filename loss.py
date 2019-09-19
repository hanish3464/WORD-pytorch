import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class WTD_LOSS(nn.Module):
    def __init__(self, use_gpu = True):
        super(WTD_LOSS, self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss
            #sum_loss += loss/average_number

        return sum_loss

    def forward(self, region_score_GT, affinity_score_GT, score_region, score_affinity, confidence):
        region_score_GT =region_score_GT
        affinity_score_GT=affinity_score_GT
        score_region =score_region
        score_affinity = score_affinity
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        #loss_fn = torch.nn.MSELoss(reduction='none')
        #assert score_region.size() == region_score_GT.size() and score_affinity.size() == affinity_score_GT.size()
        loss1 = loss_fn(score_region, region_score_GT)
        loss2 = loss_fn(score_affinity, affinity_score_GT)
        loss_g = torch.mul(loss1, confidence)
        loss_a = torch.mul(loss2, confidence)
        char_loss = self.single_image_loss(loss_g, region_score_GT)
        affi_loss = self.single_image_loss(loss_a, affinity_score_GT)
        return char_loss / loss_g.shape[0] + affi_loss / loss_a.shape[0]
