import torch
from torch import nn
from torch.nn import functional as F

### this is for running in local ###
import os
try:
    os.environ['HTTP_PROXY']='http://185.46.212.90:80'
    os.environ['HTTPS_PROXY']='http://185.46.212.90:80'
    print ("proxy_exported")
except:
    None

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        """
        Initializes the PatchEmbedding layer.
        Args:
            in_channels (int): Number of input channels.
            patch_size (int): Size of each patch.
            embedding_dim (int): Dimension of the embedding.
        """
        super().__init__()
        self.patcher = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.patch_size = patch_size

    def forward(self, x):
        if x.shape[-1] % self.patch_size != 0:
            raise ValueError(f"Input image size must be divisible by patch size {self.patch_size}")
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


# Attention Head with Improved Buffer Initialization
class AttentionHead(nn.Module):
    def __init__(self, head_size, num_embed, seq_len, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.tril = torch.tril(torch.ones(seq_len, seq_len))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        score = q @ k.transpose(-2, -1) * C**-0.5
        score = score.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        score = torch.softmax(score, dim=-1)
        score = self.dropout(score)
        val = self.value(x)
        out = score @ val
        return out

# Multi-Head Attention Decoder Block
class MultiHeadAttentionBlockDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, attn_dropout, seq_len):
        super().__init__()
        self.head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(self.head_size, embedding_dim, seq_len, attn_dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Multi-Head Attention Block
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output



# MLP Block
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

# Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiHeadAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


# Transformer Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0, seq_len=64):
        super().__init__()
        self.msa_block = MultiHeadAttentionBlockDecoder(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout, seq_len=seq_len)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class Bert(nn.Module):
    def __init__(self, n_embeddings=40000, seq_len=20, num_transformer_layers=8, embedding_dim=128,
                 mlp_size=128 * 4, num_heads=8, attn_dropout=0.1, mlp_dropout=0.1, embedding_dropout=0):
        super().__init__()
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.rand(1, seq_len, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          mlp_dropout=mlp_dropout,
                                                                          attn_dropout=attn_dropout)
                                                  for _ in range(num_transformer_layers)])
        self.output_embedding = nn.Sequential(nn.LayerNorm(embedding_dim),
                                              nn.Linear(embedding_dim, n_embeddings, bias=False))

    def forward(self, x):
        x = self.embeddings(x) + self.positional_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.output_embedding(x)
        return x


class Gpt(nn.Module):
    def __init__(self, n_embeddings=40000, seq_len=64, num_transformer_layers=6, embedding_dim=128 * 6,
                 mlp_size=128 * 4, num_heads=6, attn_dropout=0.1, mlp_dropout=0.1, embedding_dropout=0):
        super().__init__()
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.rand(1, seq_len, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_decoder = nn.Sequential(*[TransformerDecoderBlock(embedding_dim=embedding_dim,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          mlp_dropout=mlp_dropout,
                                                                          attn_dropout=attn_dropout,
                                                                          seq_len=seq_len)
                                                  for _ in range(num_transformer_layers)])
        self.output_logits = nn.Sequential(nn.LayerNorm(embedding_dim),
                                           nn.Linear(embedding_dim, n_embeddings, bias=False))

    def forward(self, x, target=None):
        x = self.embeddings(x) + self.positional_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_decoder(x)
        logits = self.output_logits(x)

        if target is not None:
            logits = logits.view(-1, logits.size(-1))
            target = target.view(-1)
            loss = nn.functional.cross_entropy(logits, target)
        else:
            loss = None
        return logits, loss


class ViT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_transformer_layers=12,
                 embedding_dim=768, mlp_size=3072, num_heads=12, attn_dropout=0, mlp_dropout=0.1,
                 embedding_dropout=0.1, num_classes=1000):
        super().__init__()
        assert img_size % patch_size == 0, "Input image size must be divisible by patch size"
        self.num_patches = (img_size * img_size) // (patch_size ** 2)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        self.positional_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, embedding_dim), requires_grad=True)
        self.class_embedding = nn.Parameter(torch.rand(1, 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          mlp_dropout=mlp_dropout,
                                                                          attn_dropout=attn_dropout)
                                                  for _ in range(num_transformer_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, num_classes))

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.size(0)
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.positional_embedding
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x















############## Self Attention Decoder Block ################
class AttentionHead(nn.Module):
    def __init__(self, head_size, num_embed, seq_len, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len))) #Assigning tril to the module using register_buffer
        self.dropout = nn.Dropout(dropout) #Adding dropout layer

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        score = q @ k.transpose(-2, -1) * C**-0.5
        score = score.masked_fill(self.tril[:T, :T] == 0, float("-inf")) #masking future positions
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        val = self.value(x) #computing weighted aggregation
        out = score @ val
        return out

################## Multi-Head Attention Decoder Block ##################
class MultiHeadAttentionBlockDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, attn_dropout, seq_len):
        super().__init__()
        self.head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(self.head_size, embedding_dim, seq_len, attn_dropout) for head_idx in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # Concatenate outputs from multiple attention heads
        out = self.dropout(self.proj(out)) # Apply linear projection and dropout
        return out


################### Multi-Head Attention Block ##################
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0, mask=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x) # Apply layer normalization
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False) # Multi-head attention operation
        return attn_output

#################### MLP Block ######################
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x) # Apply layer normalization
        x = self.mlp(x) # Multi-layer perceptron (MLP) operation
        return x

############## Transformer Encoder Block ##################
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiHeadAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x # Multi-head self-attention block
        x = self.mlp_block(x) + x # Multi-layer perceptron (MLP) block
        return x

# +
############### Transformer Decoder Block ################

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0, seq_len=64):
        super().__init__()
        self.msa_block = MultiHeadAttentionBlockDecoder(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout, seq_len=seq_len)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x # Multi-head self-attention decoder block
        x = self.mlp_block(x) + x # Multi-layer perceptron (MLP) block
        return x


# -

############## BERT Model ####################
class Bert(nn.Module):
    def __init__(self, n_embeddings=40000, seq_len=20, num_transformer_layers=8, embedding_dim=128,
                 mlp_size=128 * 4, num_heads=8, attn_dropout=0.1, mlp_dropout=0.1, embedding_dropout=0):
        super().__init__()
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        self.positional_embedding = nn.Parameter(data=torch.rand(1, seq_len, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          mlp_dropout=mlp_dropout,
                                                                          attn_dropout=attn_dropout)
                                                  for _ in range(num_transformer_layers)])
        self.output_embedding = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                              nn.Linear(in_features=embedding_dim, out_features=n_embeddings, bias=False))

    def forward(self, x):
        x = self.embeddings(x)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.output_embedding(x)
        return x


# +
################## GPT Model ##################

class Gpt(nn.Module):
    def __init__(self, n_embeddings=40000, seq_len=64, num_transformer_layers=6, embedding_dim=128 * 6,
                 mlp_size=128 * 4, num_heads=6, attn_dropout=0.1, mlp_dropout=0.1, embedding_dropout=0):
        super().__init__()
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)
        self.positional_embedding = nn.Parameter(data=torch.rand(1, seq_len, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_decoder = nn.Sequential(*[TransformerDecoderBlock(embedding_dim=embedding_dim,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          mlp_dropout=mlp_dropout,
                                                                          attn_dropout=attn_dropout,
                                                                          seq_len=seq_len)
                                                  for _ in range(num_transformer_layers)])
        self.output_logits = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                           nn.Linear(in_features=embedding_dim, out_features=n_embeddings, bias=False))

    def forward(self, x, target=None):
        B, T = x.shape
        x = self.embeddings(x)
        x = self.positional_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_decoder(x)
        logits = self.output_logits(x)

        if target is not None:
            B, T, C = logits.shape
            logits = torch.reshape(logits, (B * T, C))
            target = torch.reshape(target, (B * T,))
            loss = F.cross_entropy(logits, target)
        else:
            loss = None
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self.forward(idx_crop)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# +
############ Vision Transformer (ViT) Model ##############

class ViT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_transformer_layers=12,
                 embedding_dim=768, mlp_size=3072, num_heads=12, attn_dropout=0, mlp_dropout=0.1,
                 embedding_dropout=0.1, num_classes=1000):
        super().__init__()
        assert img_size % patch_size == 0, "Ensure input image is divisible by patch size"
        self.num_patches = (img_size * img_size) // patch_size**2
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        self.positional_embedding = nn.Parameter(data=torch.rand(1, self.num_patches + 1, embedding_dim), requires_grad=True)
        self.class_embedding = nn.Parameter(data=torch.rand(1, 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          mlp_dropout=mlp_dropout,
                                                                          attn_dropout=attn_dropout)
                                                  for layer_idx in range(num_transformer_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim),
                                        nn.Linear(in_features=embedding_dim, out_features=num_classes))

    def forward(self, x):
        batch_size = x.shape[0] # Extract class token
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x) # Patch embedding
        x = torch.cat((class_token, x), dim=1) # Concatenate class token with patches
        x = self.positional_embedding + x # Add positional embeddings and apply dropout
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x) # Transformer encoder
        x = self.classifier(x[:, 0]) # Classifier
        return x
