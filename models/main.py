from torch.utils.tensorboard import SummaryWriter
import torchmetrics.functional as f
import torchvision.transforms as tt
import torch.nn.functional as F
from models.utils import *
from tqdm import tqdm
from math import sqrt
import torch.nn as nn
import torch
import os

training_modes = {
    'pre1l1': 'Pretraining for l1 tabular data model',
    'pre2l1': 'L1 -> OMNI pretraining',
    'l1finetune': 'L1 -> Dst',
    'pre1sdo': 'Reconstruction autoencoder for 2d CNN',
    'sdofinetune': 'SDO -> Dst'
}

"""
SEQUENTIAL METHODS FOR REAL TIME GEOMAGNETIC FORECASTING

1. Multimodal Transformer architecture. (DSCOVR ,ACE, SDO) (done)
 - Monte Carlo Sampling Deep Neural Network fc
 - XGBOOST fc
 - Random Forest fc
2. Transformer architecture. (DSCOVR, ACE) (done)
 - Monte Carlo Sampling Deep Neural Network fc
 - XGBOOST fc
 - Random Forest fc
3. Multimodal residual RNN-3D CNN based model. (DSCOVR, ACE, SDO) (done)
 - Monte Carlo Sampling Deep Neural Network fc
 - XGBOOST fc
 - Random Forest fc
4. Residual RNN model. (DSCOVR, ACE) (done)
 - Monte Carlo Sampling Deep Neural Network fc
 - XGBOOST fc
 - Random Forest fc
5. 3D CNN. (SDO) (almost done)
 - Monte Carlo Sampling Deep Neural Network fc
 - XGBOOST fc
 - Random Forest fc
"""

"""
Training Phase module
It has utilities for model training shared through inheritance
"""
class TrainingPhase(nn.Module):
    def __init__(
            self,
            config: dict,
            name_run: str,
            task: str,
            num_classes: int = None
    ):
        """
        task: 'reg', 'mult', 'bin'
        config: dict of training config with optimizer, last model, and configs
        name_run: trivial
        """
        super().__init__()
        assert (task in ['reg', 'mult', 'bin']), 'task names: reg, mult, bin'
        if config is None:
            if task is None:
                raise ValueError('Without config, task parameter is mandatory')
        self.task = task
        if self.task == 'mult' and num_classes is not None:
            self.num_class = num_classes
        elif self.task == 'mult' and num_classes is None:
            raise ValueError('Ought input num_classes for multiclass task')
        elif self.task != 'mult' and num_classes is not None:
            raise ValueError('Gave num_classes parameter for non-multiclass task')
        else: 
            pass
        self.config = config
        #Tensorboard writer
        self.writer = SummaryWriter(f'{name_run}')
        self.name = name_run
        self.train_epoch = []
        self.val_epoch = []
    def batch_metrics(self, preds, targets, mode: str):
        if mode == 'Training':
            self.train()
        elif mode == 'Validation':
            self.eval()
        else:
            raise ValueError(f'{mode} is not a valid mode: [Validation, Training]')
        if self.task == 'reg':
            metrics = {
                'mse': F.mse_loss(preds, targets),
                'mae': F.l1_loss(preds, targets),
                'r2': f.r2_score(preds, targets),
            }
            self.writer.add_scalars(f'{mode}/Metrics', metrics, self.global_step_val)
        elif self.task == 'mult':
            metrics = {
                'cross_entropy': F.cross_entropy(preds, targets),
                'accuracy': f.accuracy(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
                'recall': f.recall(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
                'precision': f.precision(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
                'f1': f.f1_score(preds, targets, 'multiclass', num_classes = self.num_class, average='weighted'),
            }
            self.writer.add_scalars(f'{mode}/Metrics', metrics, self.global_step_val)
        else:
            metrics = {
                'binary_cross_entropy': F.binary_cross_entropy(preds, targets),
                'accuracy': f.accuracy(preds, targets, task = 'binary'),
                'recall': f.recall(preds, targets, task = 'binary'),
                'precision': f.precision(preds, targets, task = 'binary'),
                'f1': f.f1_score(preds, targets, task = 'binary')
            }
            self.writer.add_scalars(f'{mode}/Metrics', metrics, self.global_step_val)
        self.writer.flush()
        if mode == 'Training':
            self.train_epoch.append(metrics)
        elif mode == 'Validation':
            self.val_epoch.append(metrics)
    @torch.no_grad()
    def validation_step(
            self,
            batch
    ):
        preds = self(batch['input'])
        self.batch_metrics(preds, batch['target'], 'Validation')
        self.config['global_step_val'] += 1
    def training_step(
            self, 
            batch, 
            criterion,
            lr_sched: torch.optim.lr_scheduler,
            grad_clip: float = False,
            training_mode: str = 'l2',
            encoder_forcing: bool = True,
            weights: list = [0.8,0.2],
            lambdas: list = [0.1 for _ in range(8)]
    ):
        """
        out_l1
        out_l2
        criterion: list (encoder forcing == True)
            criterion[0]: forcing criterion
            criterion[1]: priority criterion
        criterion: criterion (encoder forcing == False)

        """
        if training_mode == 'joint_finetune':
            preds = self(batch['img'], batch['tab'])
            loss = criterion(preds, batch['dst'])
            self.writer.add_scalars('Training/Loss', {
                'Weighted class-based Loss': loss.item(),
            }, self.config['global_step_train'])
            self.batch_metrics(preds, batch['dst'], 'Training')
            self.writer.flush()
        if training_mode == 'joint_swarm':
            preds = self(batch['img'], batch['tab'])
            loss = criterion(preds, batch['swarm'], batch['dst'])
            self.writer.add_scalars('Training/Loss', {
                'Weighted class-based Loss': loss.item(),
            }, self.config['global_step_train'])
            self.batch_metrics(preds, batch['swarm'], 'Training')
            self.writer.flush()
        elif training_mode == 'img_autoencoder':
            preds = self(batch['input'])
            loss = criterion(preds, batch['target'], batch['dst'])
            self.writer.add_scalars('Training/Loss', {
                'Weighted class-based Loss': loss.item(),
            }, self.config['global_step_train'])
            self.batch_metrics(preds, batch['target'], 'Training')
            self.writer.flush()
        elif training_mode == 'l2':
            # Weighted criterion towards very noisy samples
            noise = NoiseLoss.compute(batch['l1'], batch['l2'])
            preds = self(batch['l1'])
            loss = criterion(preds, batch['l2'], noise)
            #Add into tensorboard
            self.writer.add_scalars('Training/Loss', {
                'Weighted Noise-based Loss': loss.item(),
            }, self.config['global_step_train'])
            self.batch_metrics(preds, batch['l2'], 'Training')
            self.writer.flush()
        elif training_mode =='swarm':
            if encoder_forcing:
                noise = NoiseLoss.compute(batch['l1'], batch['l2'])
                preds, forcing_loss = self.forcing_forward(batch, criterion[0], noise)
                preds = self.fc(preds)
                priority_loss = criterion[1](preds, batch['swarm'], batch['dst'])
                loss = priority_loss*weights[0] + forcing_loss*weights[1]
                self.writer.add_scalars('Training/Loss', {
                    'Forcing Loss': forcing_loss.item(),
                    'Priority Loss': priority_loss.item(),
                    'Overal Loss': loss.item(),
                }, self.config['global_step_train'])
                self.batch_metrics(preds, batch['swarm'], 'Training')
                self.writer.flush()
            else:
                preds = self(batch['l1'])
                loss = criterion(preds, batch['swarm'], batch['dst'])
        elif training_mode == 'finetune':
            preds = self(batch['input'])
            loss = criterion(batch['input'], batch['dst'])
        
        #backpropagate
        loss.backward()
        #gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(self.parameters(), grad_clip)
        #optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        #scheduler step
        if lr_sched:
            lr = get_lr(self.optimizer)
            self.writer.add_scalars('Training/Loss', {'lr': lr}, self.config['global_step_train'])
            self.scheduler.step()
        self.writer.flush()
        # Global step shift
        self.config['global_step_train'] += 1
    @torch.no_grad()
    def end_of_epoch(self) -> None:
        keys = self.train_history[0].keys() + self.validation_keys[0].keys()
        metrics = [(train_dict.values() + val_dict.values()) for train_dict, val_dict in zip(self.train_history, self.val_history)]
        metrics = torch.tensor(metrics, dtype = torch.float32)
        metrics = [value.item() for value in metrics.mean(-2)]

        self.writer.add_scalars(
            'Overall',
            {
                k:v for k,v in zip(keys, metrics)
            },
            global_step = self.config['last_epoch']
        )
        self.writer.add_hparams(
            metric_dict = {
                k:v for k,v in zip(keys, metrics)
            },
            global_step = self.config['last_epoch']
        )
        self.writer.flush()
        self.train_epoch = []
        self.val_epoch = []     
    @torch.no_grad()
    def save_config(self) -> None:
        os.makedirs(f'{self.name}/models', exists_ok = True)
        #Model weights
        self.config['optimizer_state_dict'] = self.optimizer.state_dict()
        self.config['scheduler_state_dict'] = self.scheduler.state_dict()
        self.config['model_state_dict'] = self.state_dict()

        torch.save(self.config, f'./{self.name}/models/{self.config["name"]}_{self.config["last_epoch"]}.pt')

    def fit(
            self,
            args,
            criterion: any, #list of criterions or single criterion 
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            epochs: int,
            lr: float,
            weight_decay: float = 0.,
            grad_clip: bool = False,
            opt_func: torch.optim = torch.optim.Adam,
            lr_sched: torch.optim.lr_scheduler = None,
            encoder_forcing: bool = True,
            weights: list = [0.1,0.8],
            saving_div: int = 5,

    ) -> None:
        self.writer.add_hparams(hparam_dict={
            'epochs': epochs,
            'init_learning_rate': lr,
            'batch_size': train_loader[0].values()[0].shape[0],
            'weight_decay': weight_decay,
            'grad_clip': 0 if not grad_clip else grad_clip,
            'forcing_weight': weights[0] if encoder_forcing else None,
            'priority_weight': weights[1] if encoder_forcing else 1,
            'arch/d_model': args.d_model,
            'arch/sequence_length': train_loader[0].values()[0].shape[1],
            'arch/num_heads': args.num_heads,
            'arch/num_layers': args.num_layers,
        })

        assert (encoder_forcing and weights), 'If encoder forcing -> weights are mandatory'
        assert (encoder_forcing and isinstance(criterion, list)), 'If encoder forcing -> criterion = [forcing_criterion, priority_criterion]'
        if self.config['optimizer_state_dict'] is not None:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay).load_state_dict(self.config['optimizer_state_dict'])
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            self.optimizer = opt_func(self.parameters(), lr, weight_decay=weight_decay)
        if lr_sched is not None:
            if self.config['scheduler_state_dict'] is not None:
                self.scheduler = lr_sched(self.optimizer, lr, len(train_loader)).load_state_dict(self.config['scheduler_state_dict'])
                self.scheduler.learning_rate = lr
            else:
                self.scheduler = lr_sched(self.optimizer, lr, len(train_loader))
            lr_sched = True
        # Add model to graph
        forward_input = torch.randn(*train_loader[0].shape)
        self.writer.add_graph(self,forward_input)
        del forward_input

        for epoch in range(self.config['last_epoch'], self.config['last_epoch'] + epochs):
            # decorating iterable dataloaders
            train_loader = tqdm(train_loader, desc = f'Training - Epoch: {epoch}')
            val_loader = tqdm(val_loader, desc = f'Validation - Epoch: {epoch}')
            self.train()
            for train_batch in train_loader:
                #training step
                self.training_step(train_batch, criterion, grad_clip, encoder_forcing, weights, lr_sched)
            self.eval()
            for val_batch in val_loader:
                self.validation_step(val_batch)
            #Save model and config if epoch mod(saving_div) = 0
            if epoch % saving_div == 0:
                self.save_config()
            #End of epoch
            self.end_of_epoch()
            #Next epoch
            self.config['last_epoch'] = epoch

"""
Root Mean Squared Normalization (https://arxiv.org/pdf/1910.07467.pdf)

from summary:

Extensive experiments on several tasks using diverse network architectures 
show that RMSNorm achieves comparable performanceagainst LayerNorm but reduces 
the running time by 7%-64% on different models
"""
class RootMeanSquaredNormalization(nn.Module):
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6
    ):
        super(RootMeanSquaredNormalization, self).__init__()
        self.eps = eps
        self.g_i = nn.Parameter(torch.ones(dim))

    def forward(
            self,
            x_n
    ):
        # RMSN(x_i) = g_i*(x_i/(RMSE(x_i) + eps))
        return self.g_i*(x_n * torch.rsqrt(x_n.pow(2).mean(-1, keepdim = True) + self.eps))

"""
Attention Module:

FlashAttention Implementation: (https://arxiv.org/pdf/2205.14135.pdf)

attention: Normal attention

"""   
class Attention(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
    @staticmethod
    def flashattention(queries, keys,values, mask = None, dropout = 0, scale: float = 1):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=False, 
            enable_mem_efficient=False
        ):
            out = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                mask,
                dropout,
                scale = scale
            )
        return out
    @staticmethod
    def attention(queries, keys, values, mask = None, dropout = 0, scale: float = 1):
        F.scaled_dot_product_attention(queries, keys, values, mask, dropout, scale = scale)

"""
Moded attention and other transformer utils
"""

    
class PreNormResidual(nn.Module):
  def __init__(
          self,
          norm,
          dropout: float = 0.,
  ):
      super().__init__()
      self.dropout = nn.Dropout(dropout)
      self.layer_norm = norm
  def forward(self, out, sublayer):
      return out+self.dropout(sublayer(self.layer_norm(out)))


"""
RNN build in modules
LSTM: Long short Term Memory
GRU:Gated Recurrent Unit
"""
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
    def forward(self, x, hn = None, cn = None):
        batch_size,_,_ = x.size()
        if hn is None:
            hn = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.hidden_size, requires_grad=True, device= 'cuda')
        if cn is None:
            cn = torch.zeros(self.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.hidden_size, requires_grad=True, device = 'cuda')
        out, (_,_) = self.lstm(x, (hn,cn))
        return out

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0, bidirectional = True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout = dropout, bidirectional = bidirectional, batch_first = True)
    def forward(self, x, hn = None):
        batch_size,_,_ = x.size()
        if hn is None:
            hn = torch.zeros(self.num_layers * (2 if self.gru.bidirectional else 1), batch_size, self.hidden_size, requires_grad=True)
        out, _ = self.gru(x, hn)
        return out
"""
Monte Carlo Sampling with fc layer
"""
class MonteCarloFC(nn.Module):
    def __init__(
            self,
            fc_layer,
            dropout: float = 0.2,
            n_sampling: int = 5
    ):
        super().__init__()
        self.n_sampling = n_sampling
        self.fc = fc_layer
        self.dropout = dropout
    def forward(self, x):
        outputs = []
        for _ in range(self.n_sampling):
            self.train()
            x = self.dropout(x)
            self.eval()
            outputs.append(self.fc(x))
        out = torch.mean(torch.stack(outputs, dim = 0), dim = 0)
        return out


"""
Rotary Positional Encoder [source](https://arxiv.org/pdf/2104.09864.pdf)

RoFormer (applied to transformer architecture to enhance performance)
Llama (applied to keys and queries before MultiQueryAttention)

I'll probably make my own implementation for manual headed attention with
directed variable attention analysis
"""
class RotaryPositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model: int,
            seq_len: int,
            theta: int = 10000,
            dtype = torch.float32,
            device = get_default_device(),

    ):
        super(RotaryPositionalEncoding, self).__init__()
        """
        Creating rotary transformation matrix

        Given the embedding space V, a linear space in R^n , there is the finite sucession {x_n}_{n=1}^N where N 
        is the number of samples onto a single sequence (sequence_length), where implicitly x_i /from V for i .
        We want to introduce a linear transformation onto this so it makes a learnable rotation into the 
        embedding space 
        """
        self.device = device

        #embedding size must be even
        assert (d_model% 2 == 0), 'd_model must be div by 2'
        #Create all thetas (theta_i) for i in range(0,ndim/2) theta^(-(2i)/ndim)
        theta_j = torch.tensor([1 / theta**((2*i)/d_model) for i in range(d_model/2)], dtype = dtype, device=self.device)
        #creates absolute position based on seq_len
        m_i = torch.arange(seq_len, )
        #creates (m_i,theta_j) matrix 
        function_inputs = torch.outer(m_i, theta_j)
        #translated into polar
        self.rotary_transformation = torch.polar(torch.ones_like(function_inputs), function_inputs).unsqueeze(0).unsqueeze(2)

    def forward(
            self,
            x_n: torch.tensor,
    ):
        #resampling input from embedding space into (batch_size, seq_len, embedding_size/2) 
        #(B, N, d_model) -> (B,N,d_model/2) polar transformation
        resampled_input = torch.view_as_complex(x_n.float().reshape(*x_n.shape[:-1], -1, 2))
        # F: ((1, N, 1, d_model/2), (B,N,H,d_model/2)) -> (B,N,H,d_model/2)
        rotated_batch = self.rotary_transformation * resampled_input
        # (B,N,H,d_model/2) -> (B,N,H, d_model/2, 2)
        rot_out = torch.view_as_real(rotated_batch)
        # (B,N,H,d_model/2, 2) -> (B,N,H,d_model)
        rot_out = rot_out.reshape(*x_n.shape)
        return rot_out.type_as().to(self.device)
"""
Dn Positional Encoding
Adds the first n-degree derivatives of the samples, creates lineal time dependence.
"""
class DnPositionalEncoding(nn.Module):
    def __init__(
            self,
            delta_t: timedelta,
            degree: int = 1,
            edge_order: int = 1
    ):
        super().__init__()
        self.delta_t = delta_t.total_seconds()
        self.degree = degree
        self.edge_order = edge_order

    def forward(self, x_n):
        out = x_n.clone()
        for _ in range(1,self.degree+1):
            x_n = torch.gradient(x_n, spacing = (self.delta_t, ), dim = -1, edge_order=self.edge_order)
            out += x_n
        return out


"""
MLPs
PosWiseFFN
SeqWiseFFN
"""

class PosWiseFFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            hidden_dim: int,
            activation: torch.nn = nn.SiLU,

    ):
        super(PosWiseFFN, self).__init__()
        self.activation = activation
        self.w_1 = nn.Linear(d_model, hidden_dim)
        self.v = nn.Linear(d_model, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, d_model)
    def forward(self, x_n):
        #(B,N,d_model) -> (B,N,hidden_dim) -> (B,N,d_model)
        return self.w_2(self.activation(self.w_1(x_n)) + self.v(x_n))

class SeqWiseMLP(nn.Module):
    def __init__(
            self,
            seq_len: int,
            hidden_dim: int,
            activation = nn.SiLU,
    ):
        super().__init__()
        self.activation = activation
        self.w_1 = nn.Linear(seq_len, hidden_dim)
        self.v = nn.Linear(seq_len, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, seq_len)
    def forward(
            self,
            x_n: torch.Tensor,
    ):
        super().__init__()
        x_n = x_n.tranpose(-1,-2)
        return self.w_2(self.activation(self.w_1(x_n)) + self.v(x_n)).transpose(-1,-2)

"""Transformer architecture"""

class Transformer(nn.Module):
	def __init__(
			self,
			encoder: nn.Module,
			positional_encoding: nn.Module,
			fc_layer: nn.Module,
			num_layers: int = 4,
	):
		super().__init__()
		self.pe = positional_encoding
		self.encoder = nn.TransformerEncoder(
			encoder, 
			num_layers
		)
		self.fc = fc_layer
	def forward(self, x_n):
		out = self.pe(x_n)
		out = self.encoder(out)
		out = self.fc(out)
		return out
	#Only for L1
	def forcing_forward(
			self,
			batch,
			criterion,
			noise_factor,
	) -> tuple:
		forcing_loss = 0
		l1 = batch['l1']
		l2 = batch['l2']
		for rnn, residual_connection in zip(self.rnn, self.residual_connections):
			l1 = residual_connection(l1, lambda x: rnn(x))
			l2 = residual_connection(l2, lambda x: rnn(x))
			forcing_loss += criterion(l1, l2, noise_factor)
		return l1, forcing_loss	

