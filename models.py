import torch

class VulBERTa_Vanilla(torch.nn.Module):
    def __init__(self,base_model,n_clases,base_model_output_size=768, dropout=0.1):
        super().__init__()
        
        self.num_labels = n_clases
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(768,768)
        self.fc2 = torch.nn.Linear(768,n_clases)
        
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,output_attentions=None,output_hidden_states=None,return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        x = outputs[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        #### Below is the standard output from RobertaforSequenceClassifcation head class
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VulBERTa_Extend(torch.nn.Module):
    def __init__(self,base_model,n_clases,base_model_output_size=768, dropout=0.1):
        super().__init__()
        
        self.num_labels = n_clases
        self.base_model = base_model
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(768,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,n_clases)
        
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,output_attentions=None,output_hidden_states=None,return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        x = outputs[0]
        x = x[:, 0, :]
        x = self.dropout1(x)
        x = torch.nn.functional.selu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.nn.functional.selu(self.fc2(x))
        logits = self.fc3(x)
        
        #### Below is the standard output from RobertaforSequenceClassifcation head class
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class VulBERTa_CNN(torch.nn.Module):
    def __init__(self,base_model,n_clases,base_model_output_size=768, dropout=0.2):
        super().__init__()
        
        self.num_labels = n_clases
        self.base_model = base_model
        self.dropout1 = torch.nn.Dropout(dropout)
        #self.dropout2 = torch.nn.Dropout(dropout)
        #self.fc1 = torch.nn.Linear(768,512)
        self.fc2 = torch.nn.Linear(300,128)
        self.fc3 = torch.nn.Linear(128,n_clases)
        
#        self.conv = torch.nn.Conv1d(in_channels=768, out_channels=512, kernel_size=9)
        
        self.conv1 = torch.nn.Conv1d(in_channels=768, out_channels=100, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=768, out_channels=100, kernel_size=4)
        self.conv3 = torch.nn.Conv1d(in_channels=768, out_channels=100, kernel_size=5)
        
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,output_attentions=None,output_hidden_states=None,return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        
        x = outputs[0]
        #x = x[:, 0, :]
        x = x.permute(0,2,1)
        
        
        x1 = torch.nn.functional.relu(self.conv1(x))
        x2 = torch.nn.functional.relu(self.conv2(x))
        x3 = torch.nn.functional.relu(self.conv3(x))
        
        x1 = torch.nn.functional.max_pool1d(x1, x1.shape[2])
        x2 = torch.nn.functional.max_pool1d(x2, x2.shape[2])
        x3 = torch.nn.functional.max_pool1d(x3, x3.shape[2])
        
        x = torch.cat([x1,x2,x3],dim=1)
        x = x.flatten(1)
        
#         x = torch.nn.functional.relu(self.conv(x))
#         x = torch.nn.functional.max_pool1d(x, 4)
#         x = torch.mean(x, -1)
#         x = self.dropout1(x)
        
        
        x = self.fc2(x)
        logits = self.fc3(x)
        
        #### Below is the standard output from RobertaforSequenceClassifcation head class
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class VulBERTa_LSTM(torch.nn.Module):
    def __init__(self,base_model,n_clases,base_model_output_size=768, dropout=0.2):
        super().__init__()
        
        self.num_labels = n_clases
        self.base_model = base_model
        self.dropout1 = torch.nn.Dropout(dropout)
        #self.dropout2 = torch.nn.Dropout(dropout)
        #self.fc1 = torch.nn.Linear(768,512)
        self.fc2 = torch.nn.Linear(256*2,256)
        self.fc3 = torch.nn.Linear(256,n_clases)
        
        self.lstm1 = torch.nn.LSTM(input_size=768,
                            hidden_size=256,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)        
        
    def forward(self,input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,output_attentions=None,output_hidden_states=None,return_dict=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        
        x = outputs[0]
        #x = x[:, 0, :]
        self.lstm1.flatten_parameters()
        output, (hidden, cell) = self.lstm1(x)
        x = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        logits = self.fc3(x)
        
        #### Below is the standard output from RobertaforSequenceClassifcation head class
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    
################################################## Starting line for custom pretrain

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
        
        

class RobertaLMHead(torch.nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = (torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

        self.decoder = (torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False))
        self.bias = (torch.nn.Parameter(torch.zeros(config.vocab_size)))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class RoBERTa_custom_pretrain(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config) ##### CHANGE THIS TO OUR OWN CUSTOM <---------

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    