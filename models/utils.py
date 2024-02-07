import torch.nn.functional as F
import torch.nn as nn
import torch
from torchmetrics import Metric
# Base Deep Neural Network
def  SingularLayer(input_size, output, activation):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        activation()
    )
    return out

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, architecture,hidden_activations, out_activation=None):
        super(DeepNeuralNetwork, self).__init__()
        assert (len(hidden_activations) == len(architecture)), 'Must have activation for each layer, if not, put None as activation'
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output, activation in enumerate(architecture, hidden_activations):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output, activation))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(nn.Linear(input_size, output_size))
        if out_activation is not None:
            self.output_layer.add_module(name = 'fc_layer', module = out_activation)
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out

## GPU usage

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl) 

# get learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr'] 

# Metrics
## Regression
class RootMeanSquaredError(Metric):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.add_state('mse', torch.tensor(0), dist_reduce_fx='mean')
    def update(
            self,
            preds : torch.Tensor,
            targets: torch.Tensor,
    ) -> None:
        if preds.shape != targets.shape:
            raise ValueError('preds and targets must be the same size')
        self.mse += torch.pow(preds-targets, 2)

    def compute(self) -> torch.Tensor:
        return torch.sqrt(self.mse)
        
"""
WEIGHTED LOSS FUNCTIONS
"""
## Utils
def map_to_discrete(tensor: torch.tensor) -> torch.tensor:
    thresholds = torch.tensor([-400, -250, -100, -50]) 
    discrete_values = torch.tensor([3,2,1,0])
    discrete_tensor = torch.bucketize(tensor, thresholds, right=True)
    mapped_tensor = discrete_values[discrete_tensor - 1]
    return mapped_tensor

"""
# Pretraining

targets: batch['swarm'] for all swarms
dst: batch['dst'] for weighted loss
preds: self.forward_with_forcing(batch['l1'], batch['l2'])


"""
def compute_weights(dataloader):
    bounded = torch.bincount(torch.cat([batch['dst'].view(-1) for batch in dataloader], dim = -1))
    return torch.softmax(1 / bounded)

class PretrainingPriorityCriterion(nn.Module):
    def __init__(
            self,
            dataloader
    ):
        super().__init__()
        self.base_criterion = F.mse_loss 
        # Weight calc
        self.weights = compute_weights(dataloader)
    def forward(self, pred, target, dst) -> float:
        losses = self.base_criterion(pred, target, reduction = 'none').mean(dim = (1,2))
        seq_dst, _ = torch.max(dst, dim = 1)
        return torch.mean(losses*self.weights[seq_dst])

"""
LOSS FORWARD FUNCTIONS
"""

"""
Priority Criterion

"""

class FineTuningPriorityCriterion(nn.Module):
    def __init__(
            self,
            dataloader,
    ):
        super().__init__()
        self.base_criterion = F.cross_entropy
        # Weight calc
        self.weights = compute_weights(dataloader)
    def forward(
            self,
            pred,
            target
    ) -> torch.Tensor:
        losses = self.base_criterion(pred, target, reduction = 'none').mean(dim = (1,2))
        seq_dst, _ = torch.max(target, dim = 1)
        return torch.mean(losses*self.weights[seq_dst])
        
        

"""
Forcing Criterion
Based on datasets diff
"""
def weight_scaler(dataloader):
    ## Computing weights
    batchwise_loss = []
    for batch in dataloader:
        # output -> (B)
        batchwise_loss.append(F.mse_loss(batch['l1'], batch['l2'], reduction = 'none').mean(dim = (1,2)))
    loss = torch.cat(batchwise_loss, dim = -1)
    loss_max = loss.max()
    loss_min = loss.min()
    return lambda J: (J - loss_max)/(loss_max-loss_min)

class ForcingCriterion(nn.Module):
    def __init__(
            self,
            dataloader,

    ):
        super().__init__()
        self.base_criterion = F.mse_loss
        self.weight_scaler = weight_scaler(dataloader)

    def forward(self, l1_out, l2_out, noise_loss: torch.Tensor) -> torch.Tensor:
        #B,1
        out = self.base_criterion(l1_out, l2_out, reduction = 'none').mean(dim = (1,2))
        #(B,1 * B,1).mean()
        return torch.mean(out*noise_loss.apply_(self.weight_scaler))

class NoiseLoss(nn.Module):
    @staticmethod
    def compute(l1, l2):
        # (B,S,I) x (B,S,I) -> (B,1)
        return F.mse_loss(l1, l2, reduction = 'none').mean(dim = (1,2))
