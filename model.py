from transformers import AutoModel, AlbertPreTrainedModel, AlbertModel
from transformers.modeling_albert import AlbertMLMHead

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import config_lm


class BYOLLM(AlbertPreTrainedModel):
    def __init__(self, config, mlp_hidden_size, projection_size):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.init_weights()
        self.tie_weights()

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def tie_weights(self):
        self._tie_or_clone_weights(
            self.predictions.decoder, self.albert.embeddings.word_embeddings
        )

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
        """
        I should pass only the masked token prediction  embedding to the MLP and then get it out to the loss
        """
        prediction_scores = self.mlp(prediction_scores)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
            outputs = (masked_lm_loss,) + outputs
        return outputs


"""
TODO:
1. implement any thing class by class and try in a notebook.
2. initiliaze in a notebook with one example.
"""
