import torch
import torch.nn as nn


class LTD_LOSS(nn.Module):
    def __init__(self):
        super(LTD_LOSS, self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)

        for i in range(batch_size):
            average_number = 0
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

        return sum_loss

    def forward(self, region_score_GT, affinity_score_GT, score_region, score_affinity, confidence):
        region_score_GT =region_score_GT
        affinity_score_GT=affinity_score_GT
        score_region =score_region
        score_affinity = score_affinity
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        loss1 = loss_fn(score_region, region_score_GT)
        loss2 = loss_fn(score_affinity, affinity_score_GT)
        loss_region = torch.mul(loss1, confidence)
        loss_affinity = torch.mul(loss2, confidence)
        char_loss = self.single_image_loss(loss_region, region_score_GT)
        affi_loss = self.single_image_loss(loss_affinity, affinity_score_GT)
        return char_loss / loss_region.shape[0] + affi_loss / loss_affinity.shape[0]
