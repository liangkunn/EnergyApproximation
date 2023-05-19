import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel

# grab it from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=800):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# transformerV2
class RNATransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(RNATransformer, self).__init__()

        # since the input data is already one-hot encoded, applying a linear transformation instead of nn.embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*6, dropout=dropout, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, 1)


    def forward(self, x):
        # Input shape X: (batch_size, sequence_length, input_dim)

        x = self.embedding(x)  # (batch_size, sequence_length, d_model)
        x = self.layer_norm(x)
        x = self.pos_encoder(x)  # (batch_size, sequence_length, d_model)
        x = self.dropout(x)
        x = self.transformer(x)  # (batch_size, sequence_length, d_model)
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, 1)

        return x.squeeze(-1)
    


# DistilBBert
class DistilBertForRegression(torch.nn.Module):
    def __init__(self, config, input_dim):
        super(DistilBertForRegression, self).__init__()
        self.distilbert = DistilBertModel(config)
        self.embedding = torch.nn.Linear(input_dim, config.dim)
        self.pre_classifier = torch.nn.Linear(config.dim, config.dim)
        self.classifier = torch.nn.Linear(config.dim, 1)
        self.dropout = torch.nn.Dropout(config.seq_classif_dropout)

    def forward(self, one_hot_input, attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None):
        inputs_embeds = self.embedding(one_hot_input)
        distilbert_output = self.distilbert(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = torch.nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits