BERT MODEL:

============================================================================================================================================
Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
============================================================================================================================================
Bert (Bert)                                                  [32, 20]             [32, 20, 23948]      2,560                True
├─Embedding (embeddings)                                     [32, 20]             [32, 20, 128]        3,065,344            True
├─Dropout (embedding_dropout)                                [32, 20, 128]        [32, 20, 128]        --                   --
├─Sequential (transformer_encoder)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    └─TransformerEncoderBlock (0)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (1)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (2)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (3)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (4)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (5)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (6)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
│    └─TransformerEncoderBlock (7)                           [32, 20, 128]        [32, 20, 128]        --                   True
│    │    └─MultiHeadAttentionBlock (msa_block)              [32, 20, 128]        [32, 20, 128]        66,304               True
│    │    └─MLPBlock (mlp_block)                             [32, 20, 128]        [32, 20, 128]        131,968              True
├─Sequential (output_embedding)                              [32, 20, 128]        [32, 20, 23948]      --                   True
│    └─LayerNorm (0)                                         [32, 20, 128]        [32, 20, 128]        256                  True
│    └─Linear (1)                                            [32, 20, 128]        [32, 20, 23948]      3,065,344            True
============================================================================================================================================
Total params: 7,719,680
Trainable params: 7,719,680
Non-trainable params: 0
Total mult-adds (M): 230.04
============================================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 160.62
Params size (MB): 28.75
Estimated Total Size (MB): 189.38
============================================================================================================================================




