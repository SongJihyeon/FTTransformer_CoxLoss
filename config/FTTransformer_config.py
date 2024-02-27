n_blocks = 2
activation = 'CELU'

Config = {
    'n_blocks': n_blocks, # # of Transformer blocks
    'd_block': [96, 128, 192, 256, 320, 384][n_blocks - 1], # dimension of blocks
    'attention_n_heads': 8, # Transformer multi-head attention heads
    'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35][n_blocks - 1], # Multi-head attention dropout
    
    'ffn_d_hidden': None, # Transformer FNN hidden unit
    # int(d_block * cast(float, ffn_d_hidden_multiplier))
    
    'ffn_d_hidden_multiplier': 1.3333333333333333 if activation == 'ReGLU' else 2.0, 
    # "4 / 3" for ReGLU leads to almost the same number of parameters
    # "2.0" for ReLU.
    
    'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25][n_blocks - 1], # FNN dropout
    'residual_dropout': 0.0,
    '_is_default': True # Use default optimizer? Default optimizer = AdamW
}