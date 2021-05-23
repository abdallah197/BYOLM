from transformers import AutoModel, AlbertPreTrainedModel, AlbertModel
from transformers.modeling_albert import AlbertMLMHead

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import config_lm


class ByolLanguegeModel(AlbertPreTrainedModel):
    """
    a Pytorch nn Module that incorporate BYOL approach for transformer based models.
    the model output masked tokens embeddings after being passed through an MLP layer.
    plus the logits for the model predictions and the hidden states.
    """

    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.cls = AlbertMLMHead(config)
        self.config = config
        self.mlp = MLPHead(in_channels=config.hidden_size,
                           mlp_hidden_size=config.hidden_size * 10,
                           projection_size=config.hidden_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.decoder


    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            masked_index=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            return_dict=True,
            attention_mask= attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            output_hidden_states = True
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.cls(sequence_outputs)
        """
        1: to access the hidden states
        0: to access the embedding output layer (batch_size, seq_length, hidden_size:768)
        masked_embeddings should be (batch_size, 768)
        """
        masked_embeddings = outputs.hidden_states[0]
        masked_index = masked_index.unsqueeze(1)

        masked_embeddings = torch.cat(
            [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(masked_embeddings, masked_index)])

        masked_embeddings = self.mlp(masked_embeddings.squeeze())
        return (prediction_scores, masked_embeddings, outputs.hidden_states, outputs.attentions)