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
        self.predictions = AlbertMLMHead(config)
        self.init_weights()
        self.tie_weights()
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, config.hidden_size),
        )

    def tie_weights(self):
        self._tie_or_clone_weights(
            self.predictions.decoder, self.albert.embeddings.word_embeddings
        )

    def get_output_embeddings(self):
        return self.predictions.decoder
    
    def batched_index_select(self ,input, dim, index):
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
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        masked_index = None,
        return_dict = None,
        mlp = False,
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


        output = (prediction_scores,) + outputs[2:]
        """
        1: to access the hidden states
        0: to access the embedding output layer (batch_size, seq_length, hidden_size:768)
        masked_embeddings should be (batch_size, 768)
        """
        if mlp:
            masked_embeddings = output[1][0]
            masked_index = masked_index.unsqueeze(1)
            masked_embeddings = torch.cat([ torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(masked_embeddings, masked_index) ])
            masked_embeddings = self.mlp(masked_embeddings.squeeze())
            return (masked_embeddings,) +  output[:]    
            
        return output

