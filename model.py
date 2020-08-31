from transformers import AutoModel, AlbertPreTrainedModel, AlbertModel
from transformers.modeling_albert import AlbertMLMHead

from torch import nn
import torch.nn.functional as F


from config_lm import *


def boyl_loss(x, y):
    """
    minimize the loss function 
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

"""
add mlp_hidden_size, projection_size to config
"""
class AlbertBOYL(AlbertPreTrainedModel):
    def __init__(self, config, mlp_hidden_size, projection_size):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.mlp =  MLPHead(config.hidden_size, mlp_hidden_size, projection_size)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)
        # make sure about the prediction_scores dim
        prediction_scores = self.mlp(prediction_scores)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
        if labels is not None:
            boyl_lm_loss = boyl_loss(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (boyl_lm_loss,) + outputs

        return outputs


"""
TODO:
1. implement any thing class by class and try in a notebook.
2. initiliaze in a notebook with one example.
"""
