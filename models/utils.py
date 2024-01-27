import torch.nn.functional as F
import torch.nn as nn
import torch

# Base Deep Neural Network
def  SingularLayer(input_size, output):
    out = nn.Sequential(
        nn.Linear(input_size, output),
        nn.ReLU(True)
    )
    return out

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, *args, activation=None):
        super(DeepNeuralNetwork, self).__init__()
        
        self.overall_structure = nn.Sequential()
        #Model input and hidden layer
        for num, output in enumerate(args):
            self.overall_structure.add_module(name = f'layer_{num+1}', module = SingularLayer(input_size, output))
            input_size = output

        #Model output layer
        self.output_layer = nn.Sequential(nn.Linear(input_size, output_size))
        if activation is not None:
            self.output_layer.add_module(activation)
    def forward(self, xb):
        out = self.overall_structure(xb)
        out = self.output_layer(out)
        return out

## GPU usage

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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
@torch.no_grad()
def huber_loss(y_pred, y_true, delta=1.0):
    return F.smooth_l1_loss(y_pred, y_true, reduction='mean', beta=delta)

@torch.no_grad()
def r2(y_pred, y_true):
    mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2
## Multiclass classification

@torch.no_grad()
def multiclass_accuracy(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    return torch.tensor(torch.sum(preds == target).item() / len(target))

@torch.no_grad()
def multiclass_precision(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    correct = (preds == target).float()
    true_positive = torch.sum(correct).item()
    false_positive = torch.sum(preds != target).item()
    precision = true_positive / (true_positive + false_positive + 1e-7)
    return torch.tensor(precision)

@torch.no_grad()
def multiclass_recall(predicted, target):
    _, preds = torch.max(predicted, dim=1)
    correct = (preds == target).float()
    true_positive = torch.sum(correct).item()
    false_negative = torch.sum(preds != target).item()
    recall = true_positive / (true_positive + false_negative + 1e-7)
    return torch.tensor(recall)

@torch.no_grad()
def compute_all(predictions, targets):
    accuracy = multiclass_accuracy(predictions, targets)
    precision = multiclass_precision(predictions, targets)
    recall = multiclass_recall(predictions, targets)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    return accuracy, precision, recall, f1_score


# Weighted Loss functions
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
            
class PretrainingCriterion(nn.Module):
    def __init__(
            self,
            dataloader
    ):
        super().__init__()
        self.base_criterion = F.mse_loss 
        # Weight calc
        bounded = torch.bincount(torch.cat([batch['dst'].view(-1) for batch in dataloader], dim = -1))
        self.weights = torch.softmax(1 / bounded)
    def forward(self, preds, targets, dst):
        losses = self.base_criterion(preds.view(preds.size(1)*preds.size(0), preds.size(-1)), targets.view(preds.size(1)*preds.size(0), preds.size(-1)), reduction = 'none').mean(dim = -1)
        return (losses * self.weights[dst.view(-1)]).mean()

"""
# Finetuning

targets: batch['dst']
preds: self.forward_with_forcing(batch['l1'], batch['l2'])\

"""
class FineTuningCriterion(nn.Module):
    def __init__(
            self,
            dataloader
    ):
        super().__init__()
        self.base_criterion = F.cross_entropy
        # Weight calc
        bounded = torch.bincount(map_to_discrete(torch.cat([torch.flatten(batch['dst']) for batch in dataloader], dim = -1)))
        self.weights = torch.softmax(1 / bounded)
    def forward(self, preds, targets):
        losses = self.base_criterion(preds.view(-1), targets.view(-1), reduction = 'none')
        return (losses * self.weights[targets.view(-1)]).mean()
