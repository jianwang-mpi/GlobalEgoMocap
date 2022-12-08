from torch import nn
from abc import abstractmethod
import torch


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
    
    def encode(self, input: torch.Tensor):
        raise NotImplementedError
    
    def decode(self, input: torch.Tensor):
        raise NotImplementedError
    
    def sample(self, batch_size, current_device, **kwargs) -> torch.Tensor:
        raise RuntimeWarning()
    
    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def loss_function(self, *inputs, **kwargs) -> torch.Tensor:
        pass
