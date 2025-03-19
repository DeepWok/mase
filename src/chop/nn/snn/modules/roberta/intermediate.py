import torch
import torch.nn as nn

from chop.nn.snn.modules.neuron import ANN_neuron
from chop.nn.snn.metric.averagemeter import AverageMeter

from transformers.activations import ACT2FN

class RobertaIntermediateSA(nn.Module):
    def __init__(self, config, q_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.ReLU_neuron = nn.ReLU()
        self.hidden_size = config.hidden_size
        self.hidden_states_meter = AverageMeter()
        self.ann =True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        self.hidden_states_meter.update(torch.count_nonzero(hidden_states)/torch.numel(hidden_states))
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


    def flops(self):
        flops = torch.numel(self.ReLU_neuron.max_act)*self.hidden_size
        return flops
    def flops_ANN(self):
        energy = ANN_Energy(torch.numel(self.ReLU_neuron.max_act)*self.hidden_size,torch.numel(self.ReLU_neuron.max_act),self.hidden_states_meter.avg)

        return energy

    def flops_SNN(self):
        energy = SNN_Energy(torch.numel(self.ReLU_neuron.max_act)*self.hidden_size,torch.numel(self.ReLU_neuron.max_act),self.hidden_states_meter.avg,1,self.ReLU_neuron.timstep_cycle)

        return energy
