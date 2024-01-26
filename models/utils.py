from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
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



class RegressionTrainingPhase(nn.Module):
    def __init__(
            self,
            config: dict,
            criterion = None,
            pretraining: bool = True
    ):
        super().__init__()
        self.config = config
        self.criterion = criterion()
        self.train_writer = SummaryWriter('log/train')
        self.val_writer = SummaryWriter('log/val')
        self.pretraining = pretraining
    def training_step(
            self,
            batch: torch.Tensor,
            grad_clip: float = False,
            encoder_forcing: bool = True,
            weights: list = None,
            lr_sched: bool = True,
    ) -> None:
        torch.cuda.empty_cache()
        # y_msk, x_msk
        if encoder_forcing:
            out_l1 = self.encoder(batch['l1'])
            out_l2 = self.encoder(batch['l2'])
            encoder_forcing_J = self.criterion(out_l1, out_l2, batch['dst'])*weights[0]
            if self.pretraining:
                out_l1 = self.decoder(out_l1, batch['swarm'], batch['x_msk'], batch['y_msk'])
                out_l2 = self.decoder(out_l2, batch['swarm'], batch['x_msk'], batch['y_msk'])
                encoder_forcing_J+=self.criterion(out_l1, out_l2, batch['dst'])*weights[1]
                out_l1 = self.projection(out_l1)
                main_J = self.criterion(out_l1, batch['swarm'], batch['dst'])*weights[2]
                J = main_J + encoder_forcing_J
        else:
            out = self(batch['l1'], batch['swarm'], batch['x_msk'], batch['y_msk'])
            if self.pretraining:
                J = self.criterion(out, batch['swarm'], batch['dst'])
            else:
                J = self.criterion(out, batch['dst'])

        # Include within tensorboard
        if encoder_forcing:
            self.train_writer.add_scalar('Train-Overall Loss', J.item() ,self.config['global_step_train'])
            self.train_writer.add_scalar('Train-Encoder Forcing Loss', encoder_forcing_J.item(), self.config['global_step_train'])
            self.train_writer.add_scalar('Train-Main Loss', main_J.item(), self.config['global_step_train'])
        else:
            self.train_writer.add_scalar('Train-Loss', J.item(), self.config['global_step_train'])    
        self.train_writer.flush()
        #backpropagate
        J.backward()
        #gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), grad_clip)
        #optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        #scheduler step
        if lr_sched:
            lr = get_lr(self.optimizer)
            self.train_writer.add_scalar('Learning Rate', lr, self.config['global_step_train'])
            self.train_writer.flush()
            self.scheduler.step()
        # Global step shift
        self.config['global_step_train'] += 1
    def validation_step(self, batch) -> None:
        #forward
        out = self(batch['l1'])
        if self.pretraining:
            #loss
            J = self.criterion(out, batch['swarm'])
            # r2 score
            r2_score = r2(out, batch['swarm'])
            # Mean Absolute Error
            mae = F.l1_loss(out, batch['swarm'])
            # Huber loss
            hub_loss = huber_loss(out, batch['swarm'])
            # Include within tensorboard
            self.val_writer.add_scalar('Validation-Loss', J.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-R^2 score', r2_score.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-Mean Absolute Error', mae.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-Huber Loss', hub_loss.item(), self.config['global_step_val'])
            self.val_writer.flush()
        else:
            #loss
            J = self.criterion(out, batch['dst'])
            # r2 score
            accuracy, precision, recall, f1_score = compute_all(out, batch['dst'])
            # Include within tensorboard
            self.val_writer.add_scalar('Validation-Loss', J.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-Accuracy', accuracy.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-Precision', precision.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-Recall', recall.item(), self.config['global_step_val'])
            self.val_writer.add_scalar('Validation-F1 Score', f1_score.item(), self.config['global_step_val'])
            self.val_writer.flush()

        # Validation global step
        self.config['global_step_val'] += 1

    @torch.no_grad()
    def evaluate(self, val_loader) -> None:
        for batch in val_loader:
            self.validation_step(batch)
    
    @torch.no_grad()
    def save_config(self) -> None:
        #Model weights
        self.config['optimizer_state_dict'] = self.optimizer.state_dict()
        self.config['scheduler_state_dict'] = self.scheduler.state_dict()
        self.config['model_state_dict'] = self.state_dict()

        torch.save(self.config, f'{self.config["name"]}_{self.config["last_epoch"]}.pt')

    def fit(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            lr: float,
            weight_decay: float = 0.,
            grad_clip: bool = False,
            opt_func: torch.optim = torch.optim.Adam,
            lr_sched: torch.optim.lr_scheduler = None,
            encoder_forcing: bool = True,
            weights: list = [0.2,0.8]

    ) -> None:
        self.train_writer.add_hparams(hparam_dict={
            'name': self.config['name'],
            'pretraining': self.pretraining, 
            'init_learning_rate': lr,
            'weight_decay': weight_decay,
            'grad_clip': grad_clip,
            'weights': weights,
            'encoder_forcing': True,
            'batch_size': train_loader[0].shape[0]
        })
        assert (encoder_forcing and weights), 'If encoder forcing -> weights are mandatory'
        if self.config['optimizer_state_dict'] is not None:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay).load_state_dict(self.config['optimizer_state_dict'])
        else:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        if lr_sched is not None:
            if self.config['scheduler_state_dict'] is not None:
                self.scheduler = lr_sched(self.optimizer, lr, len(train_loader)).load_state_dict(self.config['scheduler_state_dict'])
            else:
                self.scheduler = lr_sched(self.optimizer, lr, len(train_loader))
            lr_sched = True
        for epoch in range(self.config['last_epoch'], self.config['epochs']):
            # decorating iterable dataloaders
            train_loader = tqdm(train_loader, desc = f'Training - Epoch: {epoch}')
            val_loader = tqdm(val_loader, desc = f'Validation - Epoch: {epoch}')
            for batch in train_loader:
                #training step
                self.training_step(batch, grad_clip, encoder_forcing, weights, lr_sched)
            #Validation step
            self.evaluate(val_loader)
            #Next epoch
            self.config['last_epoch'] = epoch
            #Save model and config
            self.save_config()

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
