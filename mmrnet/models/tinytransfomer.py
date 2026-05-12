import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_mean


class TinyTransformer(nn.Module):
    def __init__(self, info=None):
        super(TinyTransformer, self).__init__()
        self.num_classes = info.get('num_classes')
        x_features = info.get('num_features', 3)  # Input feature dimensions
        d_model = 76 # Dimension of the model
        nhead = 4 # Number of attention heads
        num_layers = 2 # Number of transformer layers
        dim_feedforward = 132 # dimension of hidden FFN layer
        dropout = 0.1 # Dropout rate
        if self.num_classes == 5: #radhar
            self.N_max = 1605 
        elif self.num_classes == 12: #MIRAGE
            self.N_max = 348  
        else:
            raise ValueError("N_max undefined for this task")

        # Input embedding
        self.input_embedding = nn.Linear(x_features, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.mlp = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(d_model, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes)
            )

    def forward(self, data):
        if isinstance(data, Data):
            # Handle geometric Data and DataBatch types
            x = data.x
            batch = data.batch
            batch_training = True
        else:
            # Handle regular tensor input for summary
            x = data
            x = x.squeeze(0) # for ptflops
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            batch_training = False

        B = batch[-1] + 1

        # Input embedding
        x = self.input_embedding(x)

        # Transformer encoder
        # Reshape x to (batch_size, seq_len, d_model) for transformer
        if batch_training:
            unique_batches = torch.unique(batch)
            outputs = torch.zeros((B, self.N_max, x.size(1)), device=x.device)
            for b in unique_batches:
                mask = batch == b
                outputs[b, :min(mask.sum(), self.N_max)] = x[mask, :][:self.N_max] # drop the last points if stack size > N_max
            x = self.transformer_encoder(outputs) # apply transformer on the batched padded inputs (B,N_max,F)

        else:
            unique_batches = torch.unique(batch)
            outputs = []
            for b in unique_batches:
                mask = batch == b
                batch_x = x[mask]
                batch_x = self.transformer_encoder(batch_x.unsqueeze(0))  # Add batch dimension
                outputs.append(batch_x.squeeze(0))
            x = torch.cat(outputs, dim=0)


        if batch_training:
            x = torch.mean(x, dim=1)
        else:
            x = scatter_mean(x, batch, dim=0)

        # Output layers
        y = self.mlp(x)

        return y