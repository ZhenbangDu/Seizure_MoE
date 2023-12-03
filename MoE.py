# Mix-MoE for Seizure Subtype Classification.
# See "Mixture of Experts for EEG-Based Seizure Subtype Classification"
#
# Author: Zhenbang Du
#
# The code is based on the MoE implementation:
# https://github.com/davidmrau/mixture-of-experts


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from model import EEGNet
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier

class MLP(nn.Module):
    '''
    The implementation of experts.
    '''
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

class Mix_MoE(nn.Module):

    """Call a mixture of experts layer with 2-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    manu_dim: an integer - size of the manual features
    base_model: a sklearn class - the Exp0 ('None' for Seizure-MoE)
    backbone: a str - the backbone to extract features
    noisy_gating: a boolean
    """

    def __init__(self, input_size, output_size, num_experts, hidden_size, manu_dim, base_model = None, backbone = 'eegnet', noisy_gating = False):
        super(Mix_MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.manu_dim = manu_dim

        # backbone
        if backbone == 'eegnet':
            self.extractor = EEGNet()
            self.input_size = 256
        else:
            raise ValueError("Unkown network!")

        # Exp0 ('None' for Seizure-MoE)
        self.tra_expert = base_model
        self.d_experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])

        if self.tra_expert:
            self.w_gate = nn.Parameter(torch.zeros(self.input_size + self.manu_dim, num_experts + 1), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size + self.manu_dim, num_experts + 1), requires_grad=True)
            self.bn = torch.nn.BatchNorm1d(input_size + self.manu_dim, eps=1e-05)
        else:
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, num_experts), requires_grad=True)
            self.w_noise = nn.Parameter(torch.zeros(self.input_size, num_experts), requires_grad=True)
  
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.num_experts
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, noise_epsilon = 1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        # clean_logits = self.w_gate(x)
        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(self.num_experts, dim=1)
        top_k_logits = top_logits[:, :self.num_experts]
        top_k_indices = top_indices[:, :self.num_experts]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef = 1.0, return_gates = False, pretrain = False):
        """Args:
        x: tensor shape [batch_size, input_size]
        loss_coef: a scalar - multiplier on load-balancing losses
        return_gates: a boolean 
        pretrian: a boolean

        Returns:
        y: a tensor with shape [batch_size, output_size].
        loss: a scalar. 
        """
        if self.tra_expert:
            [eeg, manu] = x
            if pretrain:
                expert_outputs = [self.d_experts[i](self.extractor(eeg[i], return_feature=True)) for i in range(self.num_experts)]
                return expert_outputs
            else:
                manu = manu.squeeze()
                if len(manu.size())==1:
                    manu = manu.unsqueeze(0)
                eeg_fea = self.extractor(eeg, return_feature=True)
                mixfeature = self.bn(torch.cat([eeg_fea, manu], dim=1))
                gates, load = self.noisy_top_k_gating(mixfeature)
                importance = gates.sum(0)
                loss = self.cv_squared(importance) + self.cv_squared(load)

                expert_outputs_d = [ ]
                expert_outputs_t = [ ]
                
                for i in range(self.num_experts):
                    if eeg_fea.size()[0]:
                        result = self.d_experts[i](eeg_fea)
                        expert_outputs_d.append(result.unsqueeze(1))
                
                if isinstance(self.tra_expert, RidgeClassifier):
                    result = self.softmax(torch.tensor(self.tra_expert.decision_function(manu.detach().cpu().numpy())))
                else:
                    result = torch.tensor(self.tra_expert.predict_proba(manu.detach().cpu().numpy()))
                expert_outputs_t.append(result.unsqueeze(1))
                
                expert_outputs_d = torch.cat(expert_outputs_d, 1).to(eeg.device)
                expert_outputs_t = torch.cat(expert_outputs_t, 1).to(eeg.device)

                kl=0
                for i in range(self.num_experts):
                    kl += F.kl_div(expert_outputs_d[:, i].log_softmax(dim=-1).float(), torch.squeeze(expert_outputs_t).softmax(dim=-1).float(), reduction='batchmean')
                loss += (kl.to(torch.float32) / self.num_experts)

                expert_outputs = torch.cat([expert_outputs_d, expert_outputs_t], 1).to(eeg.device)
                y = torch.einsum('ijk, ij-> ik', expert_outputs.to(torch.float32), gates.to(torch.float32))

                if return_gates:
                    return y, loss, gates
                else:
                    return y, loss
        else:
            x = self.extractor(x, return_feature=True)
            gates, load = self.noisy_top_k_gating(x)
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef

            expert_outputs = [self.d_experts[i](x).unsqueeze(1) for i in range(self.num_experts)]
            expert_outputs = torch.cat(expert_outputs, 1)
            y = torch.einsum('ijk, ij-> ik', expert_outputs.to(torch.float32), gates.to(torch.float32))
            
            if return_gates:
                return y, loss, gates
            else:
                return y, loss