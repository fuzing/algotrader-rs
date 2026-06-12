import torch
import torch.nn as nn
import math

class LiTPatchEmbedding(nn.Module):
    """
    Transforms the raw LOB grid into patch embeddings.
    Input shape: (Batch, Channels=2, LOB_Depth, Window_Size)
    - Channels: 0 = Prices, 1 = Volumes
    - LOB_Depth: Typically 10 or 40 levels
    - Window_Size: Number of chronological history snapshots
    """
    def __init__(self, lob_depth, window_size, patch_size, d_model):
        super(LiTPatchEmbedding, self).__init__()
        self.patch_size = patch_size

        # Calculate projection size of a flattened patch (2 channels * patch_height * patch_width)
        # Assuming patch covers full LOB depth (spatial dimension) for structural integrity
        self.patch_dim = 2 * lob_depth * patch_size
        self.num_patches = window_size // patch_size

        # Linear Projection layer mapping flattened patches to the Transformer dimension
        self.projection = nn.Linear(self.patch_dim, d_model)

        # Class token to aggregate comprehensive information (akin to ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable Position Embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x shape: (B, C, H, W) -> where H = depth, W = sequence windows
        B, C, H, W = x.shape

        # Reshape to construct patches along the time window dimension
        # (B, C, H, num_patches, patch_size) -> permute to isolate patches
        x = x.view(B, C, H, self.num_patches, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4).contiguous() # (B, num_patches, C, H, patch_size)
        x = x.view(B, self.num_patches, -1)       # Flatten to (B, num_patches, patch_dim)

        # Linearly project to d_model
        x = self.projection(x) # (B, num_patches, d_model)

        # Prepend the class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches + 1, d_model)

        # Inject positional markers
        x = x + self.pos_embedding
        return x


class LimitOrderBookTransformer(nn.Module):
    """
    The official convolution-free LiT model architecture.
    Combines Patch Embeddings, Transformer Blocks, and an LSTM Temporal Head.
    """
    def __init__(self, lob_depth=10, window_size=100, patch_size=5,
                 d_model=64, nhead=4, num_transformer_layers=3,
                 lstm_hidden_dim=64, num_lstm_layers=1, num_classes=3):
        super(LimitOrderBookTransformer, self).__init__()

        # 1. Patch Generation and Linear Projection Layer
        self.patch_embed = LiTPatchEmbedding(
            lob_depth=lob_depth,
            window_size=window_size,
            patch_size=patch_size,
            d_model=d_model
        )

        # 2. Pure Self-Attention Architecture (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 3. Dynamic Temporal Modeling Head (LSTM Layers)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.1 if num_lstm_layers > 1 else 0.0
        )

        # 4. Final Classification Projection
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, num_classes)
        )

    def forward(self, x):
        # Step 1: Extract non-overlapping patch chunks and add positions
        x = self.patch_embed(x) # Output shape: (Batch, Num_Patches + 1, d_model)

        # Step 2: Feed tokens sequence into the Multi-Head Attention blocks
        x = self.transformer(x) # Output shape: (Batch, Num_Patches + 1, d_model)

        # Step 3: Run the temporal processing sequence via LSTM
        # Passing the entire transformed token landscape into the Recurrent layers
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Step 4: Isolate the last recurrent vector state for classification
        last_time_step = lstm_out[:, -1, :] # Output shape: (Batch, lstm_hidden_dim)

        # Step 5: Output logits map (Typically Down, Stationary, Up targets)
        logits = self.mlp_head(last_time_step)
        return logits


# --- Verification Pipeline Execution ---
if __name__ == '__main__':
    # Initialize mock batch size, LOB dimensions, and sequential histories
    batch_size = 16
    lob_levels = 10     # e.g., Top 10 levels of Bid/Ask matrix
    history_window = 100 # Lookback parameter sequence length
    channels = 2        # Index 0: Prices, Index 1: Volumes

    # Generate structured pseudo-LOB Tensor: (Batch, Channels, Depth, Window)
    mock_lob_tensor = torch.randn(batch_size, channels, lob_levels, history_window)
    print(f"Input LOB Matrix shape: {mock_lob_tensor.shape}")

    # Initialize the LiT Framework
    model = LimitOrderBookTransformer(
        lob_depth=lob_levels,
        window_size=history_window,
        patch_size=5,          # Split history into groups of 5 ticks
        d_model=128,           # Transformer hidden representation length
        nhead=8,               # Multi-head attention streams
        num_transformer_layers=4,
        lstm_hidden_dim=64,
        num_classes=3          # Class targets: [0: Down, 1: Flat, 2: Up]
    )

    # Compute Forward Propagation Pass
    output_logits = model(mock_lob_tensor)
    print(f"Output Predictive Logits shape: {output_logits.shape}")


