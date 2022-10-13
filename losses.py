import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class sop_trans_mat_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, Tr=0, ratio_balance=0):
        super(sop_trans_mat_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp
        self.Tr = Tr

        self.ratio_balance = ratio_balance

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))
        self.trans = nn.Parameter(torch.eye(num_classes, num_classes, dtype=torch.float32))
        self.T_balance = torch.eye(num_classes, num_classes, dtype=torch.float32)
        self.ratio_consistency = 0.9

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, output, output2, label, d1_label):
        eps = 1e-4
        T = self.trans ** 2
        T_balance = self.T_balance

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)
        T = torch.clamp(T, 0, 1)

        E = U_square - V_square

        self.E = E

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction @ (T * self.Tr + ((1 - self.Tr) * T_balance.cuda())) + U_square - V_square.detach(),
            min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)

        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot @ T + U_square - V_square), label, reduction='sum') / len(label)

        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))

        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * balance_kl

        return loss


    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)



class sop_trans_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, Tr=0, ratio_balance=0):
        super(sop_trans_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.num_examp = num_examp
        self.Tr = Tr
        self.ratio_consistency = 0.2
        self.ratio_balance = ratio_balance

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))
        self.trans = nn.Parameter(torch.empty(num_classes, num_classes, dtype=torch.float32))
        self.T_balance = torch.ones(num_classes, num_classes, dtype=torch.float32)

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)
        torch.nn.init.normal_(self.trans, mean=1, std=0)

    def forward(self, index, output, output2, label, d1_label):

        eps = 1e-4
        T = self.trans[d1_label] ** 2
        T_balance = self.T_balance[d1_label]

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)
        T = torch.clamp(T, 0 ,1)

        E = U_square - V_square

        self.E = E

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(original_prediction * (T*self.Tr+((1-self.Tr)*T_balance.cuda())) + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)

        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot * T + U_square - V_square), label, reduction='sum') / len(label)

        loss = torch.mean(-torch.sum((label) * torch.log(prediction), dim=-1))

        loss += MSE_loss

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(prediction, dim=0)
            prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            loss += self.ratio_balance * balance_kl

        return loss


    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)
