import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, p=2, **kwargs):
        self.max_norm = max_norm
        self.p = p
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=self.p, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def CalculateOutSize(self, model, channels, samples):
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def CalculateOutSize2(self, model, channels, samples):
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data)
        out = self.avg_pool_2(out)
        out = out.view(out.size()[0], -1).shape
        return out[1]

    def __init__(self, n_class=4, channels=20, samples=512, dropoutRate=0.1, kernel_length=64, kernel_length2=16, F1=8, F2=16, D=2,
                 feature_size=256):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_class = n_class
        self.channels = channels
        self.dropoutRate = dropoutRate
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2

        self.Conv2d_1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), stride=1, padding=(0, self.kernel_length // 2), bias=False)  # 'same'
        self.BatchNorm_1_1 = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.Depthwise_Conv2d = Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), stride=1, max_norm=1, p=2, groups=self.F1,
                                                     bias=False)
        self.BatchNorm_1_2 = nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3)
        self.avg_pool_1 = nn.AvgPool2d((1, 4), stride=4)
        self.Dropout_1 = nn.Dropout(p=self.dropoutRate)

        self.Separable_Conv2d = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_length2), padding=(0, self.kernel_length2 // 2),
                                          bias=False, groups=self.F1 * self.D)  # 'same'
        self.Pointwise_Conv2d = nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), bias=False)
        self.BatchNorm_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.avg_pool_2 = nn.AvgPool2d((1, 8), stride=8)

        self.fea_model = nn.Sequential(self.Conv2d_1,
                                       self.BatchNorm_1_1,
                                       self.Depthwise_Conv2d,
                                       self.BatchNorm_1_2,
                                       nn.ELU(),
                                       self.avg_pool_1,
                                       self.Dropout_1,
                                       self.Separable_Conv2d,
                                       self.Pointwise_Conv2d,
                                       self.BatchNorm_2,
                                       nn.ELU()
                                       )
        
        outsize = self.CalculateOutSize2(self.fea_model, self.channels, self.samples)
        self.classifierBlock = nn.Linear(outsize, self.n_class)
        self.Dropout = nn.Dropout(p=0.5)

    def forward(self, data, return_feature=False):
        conv_data = self.fea_model(data.permute(0, 2, 1, 3))
        conv_data = self.avg_pool_2(conv_data)
        conv_data = self.Dropout(conv_data)
        flatten_data = conv_data.view(conv_data.size()[0], -1)
        if return_feature:
            return flatten_data
        pred_label = self.classifierBlock(flatten_data)
        return pred_label, None


class Ensemble(nn.Module):
    def __init__(self, n_estimators, method='fusion'):
        super(Ensemble, self).__init__()
        self.n_estimators = n_estimators
        self.method = method
        self.estimator_ = nn.ModuleList()

    def __len__(self):
        return len(self.estimator_)

    def __getitem__(self, index):
        return self.estimator_[index]

    def add(self, node):
        node.eval()
        self.estimator_.append(node)

    def forward(self, x):
        outputs = [
                estimator(x)[0] for estimator in self.estimator_
            ]
        proba = sum(outputs) / len(outputs)

        return proba, None
