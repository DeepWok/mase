import torch
import torch.nn as nn

from chop.nn.snn.modules.neuron import ANN_neuron
from chop.nn.snn.metric.averagemeter import AverageMeter

class RobertaSelfOutputSA(nn.Module):
    def __init__(self, config, q_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.neuron_1_if = ANN_neuron()
        self.neuron_2_if = ANN_neuron()
        self.hidden_size = config.hidden_size
        self.hidden_states_meter = AverageMeter()
        self.ann = True


    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        
        self.hidden_states_meter.update(torch.count_nonzero(hidden_states)/torch.numel(hidden_states))
        
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.neuron_1_if(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.neuron_2_if(hidden_states)
        return hidden_states

    def flops(self):
        flops = torch.numel(self.neuron_1_if.max_act)*self.hidden_size ##dense
        flops += torch.numel(self.neuron_2_if.max_act) ##power norm

        return flops

    def flops_ANN(self):
        energy = ANN_Energy(torch.numel(self.neuron_1_if.max_act)*self.hidden_size,torch.numel(self.neuron_1_if.max_act),self.hidden_states_meter.avg)
        energy += ANN_Energy( torch.numel(self.neuron_2_if.max_act), torch.numel(self.neuron_2_if.max_act),self.neuron_1_if.sparisty_meter.avg)

        return energy

    def flops_SNN(self):
        energy = SNN_Energy(torch.numel(self.neuron_1_if.max_act)*self.hidden_size,torch.numel(self.neuron_1_if.max_act),self.hidden_states_meter.avg,1,self.neuron_1_if.timstep_cycle)
        energy += SNN_Energy( torch.numel(self.neuron_2_if.max_act), torch.numel(self.neuron_2_if.max_act),self.neuron_1_if.sparisty_meter.avg,1,self.neuron_2_if.timstep_cycle)

        return energy



# Copied from transformers.models.bert.modeling_bert.BertOutput
class RobertaOutputSA(nn.Module):
    def __init__(self, config, q_config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.intermediate_size= config.intermediate_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.neuron_1_if = ANN_neuron()
        self.neuron_2_if = ANN_neuron()

        self.hidden_states_meter = AverageMeter()
        self.ann =True

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        self.hidden_states_meter.update(torch.count_nonzero(hidden_states)/torch.numel(hidden_states))
        hidden_states = self.neuron_1_if(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.neuron_2_if(hidden_states)
        return hidden_states

    def flops(self):
        flops = torch.numel(self.neuron_1_if.max_act)*self.intermediate_size ##dense
        flops += torch.numel(self.neuron_2_if.max_act)

        return flops
    def flops_ANN(self):
        energy = ANN_Energy(torch.numel(self.neuron_1_if.max_act)*self.intermediate_size ,torch.numel(self.neuron_1_if.max_act),self.hidden_states_meter.avg)
        energy += ANN_Energy( torch.numel(self.neuron_2_if.max_act), torch.numel(self.neuron_2_if.max_act),self.neuron_1_if.sparisty_meter.avg)

        return energy
    def flops_SNN(self):
        energy = SNN_Energy(torch.numel(self.neuron_1_if.max_act)*self.intermediate_size ,torch.numel(self.neuron_1_if.max_act),self.hidden_states_meter.avg,1,self.neuron_1_if.timstep_cycle)
        energy += SNN_Energy( torch.numel(self.neuron_2_if.max_act), torch.numel(self.neuron_2_if.max_act),self.neuron_1_if.sparisty_meter.avg,1,self.neuron_2_if.timstep_cycle)

        return energy