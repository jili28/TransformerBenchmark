"""
Here we can store the hyperparameter configurations for each model
"""

from config import config

params = {

    'Acceptor': {
        'lr': 1e-4,
        'epochs': 200,
        #backbone
        'backbone': 'None',
        'timestamps': 512,
        'in_channels': 3,
        'out_channels': 3, #backbone output channel
        'kernel_size' : 32,
        'nb_filters' : 64,
        'use_residual': False,
        'backbone_depth' : 4,
        #transformer
        'hidden_dim' : 32,
        'dropout' : False,
        'nheads': 8,
        'dim_feedforward' : 45,
        'enc_layers' : 5,
        'dec_layers' : 5,
        'pre_norm': False,
        #model
        'device': None,
        'position_embedding': 'sine',
        'num_queries': 1,
        'maxpools': [6, 4, 4, 2],
        #data
        'word_length':512,
        'len':10000,
        'leq':True,
        'batch_size':64,
        'k': 10,
        'M': 1

    },
    'Encoder': {
        'lr': 1e-4,
        'epochs': 50,
        #model
        'layers' : 2,
        'heads' : 1,
        'd_model': 16,
        'd_ffnn': 32,
        'scaled': True,
        'eps': 1e-5,
        #data
        'word_length':1500,
        'len':32,
        'leq':True,
        'batch_size':32

    },

    'CausalEncoder': {
        'lr': 1e-4,
        'epochs': 50,
        #model
        'num_heads' : 4,
        'input_size' : 0,
        'hidden_size' : 48,
        'num_layers' : 3,

        'embedding_type' : "pw",
        #data
        'word_length': 512,
        'len':10000,
        'leq':True,
        'batch_size':32,
        'M' : 2,
        'k' : 5
    },
}
