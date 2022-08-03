from torch import nn
import torch
from .modules import Pad_Pool, Pad_Conv
import logging


class ID(nn.Module):
    """
    This class defines all the common functionality for convolutional nets
    Inherit from this class and only implement _module() and _get_nb_features_output_layer() methods
    Modules are then stacked in the forward() pass of the model
    """

    def __init__(self, input_shape, output_shape, kernel_size=32, nb_filters=32, batch_size=64, use_residual=True, depth=12, maxpools = []):
        """
        We define the layers of the network in the __init__ function
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.seq_len = self.input_shape[0]
        self.nb_channels = self.input_shape[1]
        self.depth = depth
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.use_residual = False #use_residual
        self.batch_size= batch_size
        self.nb_features = 1

        # Define all the convolutional and shortcut modules that we will need in the model
        #     logging.info(f"Using default pool {self.depth * (4,)}")
        #     self.max_pool = nn.ModuleList([nn.MaxPool1d(4, stride = 4) for i in range(self.depth)])
        logging.info(f"Number of backbone parameters: 0")
        logging.info(
            f"Number of trainable backbone parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        logging.info('--------------- use residual : ' + str(self.use_residual))
        logging.info('--------------- depth        : ' + str(self.depth))
        logging.info('--------------- kernel size  : ' + str(self.kernel_size))
        logging.info('--------------- nb filters   : ' + str(self.nb_filters))

    def forward(self, x):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are used if specified.
        """
        return x


