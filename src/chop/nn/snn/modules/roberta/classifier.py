import torch
import torch.nn as nn

from chop.nn.snn.modules.neuron import ANN_neuron

class RobertaClassificationHeadSA(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, q_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.final_if = ANN_neuron()

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.final_if(x)
        return x