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

"""
SEQUENTIAL METHODS FOR REAL TIME GEOMAGNETIC FORECASTING

1. Multimodal Transformer architecture. (DSCOVR ,ACE, SDO) (done)
 - Linear fc
 - XGBOOST fc
 - Random Forest fc
2. Transformer architecture. (DSCOVR, ACE) (done)
 - Linear fc
 - XGBOOST fc
 - Random Forest fc
3. Multimodal residual RNN-3D CNN based model. (DSCOVR, ACE, SDO) (done)
 - Linear fc
 - XGBOOST fc
 - Random Forest fc
4. Residual RNN model. (DSCOVR, ACE) (done)
 - Linear fc
 - XGBOOST fc
 - Random Forest fc
5. 3D CNN. (SDO) (almost done)
 - Linear fc
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
    @torch.nograd
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
            encoder_forcing: float = True,
            weights: list = [0.1,0.9]


    ):
        """
        out_l1
        out_l2
        criterion: list (encoder forcing == True)
            criterion[0]: forcing criterion
            criterion[1]: priority criterion
        criterion: criterion (encoder forcing == False)

        """
        if encoder_forcing:
            #computing noise
            noise = NoiseLoss.compute(batch['l1'], batch['l2'])
            #Computing forcing loss and features
            features, forcing_loss = self.forcing_forward(batch, criterion[0], noise) 
            #to fc layer to get the prediction
            preds = self.fc(features)         
            if self.pretraining:
                priority_loss = criterion[1](preds, batch['target'], batch['dst'])
            else:
                priority_loss = criterion[1](preds, batch['target'])
            loss = forcing_loss*weights[0] + priority_loss*weights[1]
            #Add into tensorboard
            self.writer.add_scalars('Training/Loss', {
                'Weighted Encoder Forcing Loss': forcing_loss.item(),
                'Weighted Ouptut Loss': priority_loss.item(),
                'Overall Loss': loss.item()
            }, self.config['global_step_train'])
        else:
            preds = self(batch)
            if self.pretraining:
                loss = criterion(preds, batch['target'], batch['dst']) #ONLY ONE CRITERION
            else:
                loss = criterion(preds, batch['target'])
        self.writer.add_scalars('Training/Loss', {'Weighted Loss': loss} ,self.config['global_step_train'])
        self.batch_metrics(preds, batch['target'], 'Training')
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
    @torch.no_grad
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


class ModedTimeSeriesAttention(nn.Module):
  def __init__(
    self,
    d_model: int,
    num_heads: int
  ):
    super().__init__()
    """
    attention = (concat({head_i}_{i=1}^{num_heads})W_fc)^T
    head_i = softmax(\frac{Q_{i}^{T}K}{\sqrt(d_{k})})V^{T}
    """
    # Premises
    assert (d_model%num_heads == 0), 'k = \frac{hiden_dim}{num_heads} | k \in \mathbb(Z)^{+}'
    self.num_heads = num_heads
    self.d_model = d_model
    self.head_dim = d_model/num_heads
    # Projections into hidden
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    # fc projection
    self.W_fc = nn.Linear(self.head_dim*num_heads, d_model)
  def forward(self, queries, keys, values, mask = None):
    #Dimensions
    B,S,_ = queries.shape
    #Linear projection
    Q = self.W_q(queries)
    K = self.W_k(keys)
    V = self.W_v(values)
    # (B,S,d_model) ->(B,S,num_heads, head_dim)

    # Q = Q^T, K = K^T, V = V^T so that attention = soft(Q^T K / sqrt(d_k))V^T -> (B, num_heads, head_dim, seq_len)
    Q_time = Q.view(B,S,self.num_heads, self.head_dim).transpose(1,2).transpose(-1,-2)
    K_time = K.view(B,S,self.num_heads, self.head_dim).transpose(1,2).transpose(-1,-2) #SOLVE
    V_time = V.view(B,S,self.num_heads, self.head_dim).transpose(1,2).transpose(-1,-2)

    # (B,num_heads, head_dim, S)-> (B,num_heads, S, head_dim) -> (B,S,num_heads, head_dim)
    out_time = Attention.flashattention(Q_time,K_time,V_time, scale = sqrt(self.d_model)).view(B,S,self.num_heads*self.head_dim)

    # (B,S,num_heads, head_dim) -> (B,S,hidden_dim)
    stan_out = Attention.flashattention(Q,K,V, mask, scale = sqrt(self.d_model)).view(B,S,self.num_heads*self.head_dim)

    out = out_time + stan_out

    return self.W_fc(out)
    
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
Image Analysis
"""

class FeatureExtractorResnet50(nn.Module):
    def __init__(self, hidden_state_size: int, architecture: tuple, hidden_activations: tuple):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights

        self.model = resnet50(weights = ResNet50_Weights)
        
        self.model.fc = DeepNeuralNetwork(1280, hidden_state_size, architecture, hidden_activations)
        
        self.transform = tt.Compose([tt.ToTensor(), ResNet50_Weights.IMAGENET1K_V2.transforms(antialias = True)])
    def forward(self, x):
        out = self.model(x)
        return out

class FeatureExtractorVGG19(nn.Module):
    def __init__(self, hidden_state_size: int, architecture: tuple, hidden_activations: tuple):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights

        self.model = vgg19(weights = VGG19_Weights)

        self.model.classifier = DeepNeuralNetwork(1280, hidden_state_size, architecture, hidden_activations)
        
        self.transform = tt.Compose([tt.ToTensor(), VGG19_Weights.IMAGENET1K_V1.transforms(antialias = True)])
    def forward(self, x):
        out = self.model(x)
        return out
    
class GeoVideo(nn.Module):
    def __init__(self, image_model, rnn):
        super().__init__()
        self.feature_extractor = image_model
        self.sequential_extractor = rnn
    def forward(self, x):
        _, seq_length, _,_,_ = x.size()
        feature_extraction = []
        
        #feature extraction
        for t in range(seq_length):
            feature_extraction.append(self.feature_extractor(x[:,t,:,:,:]))
        
        out = torch.cat(feature_extraction)
        #sequential analysis
        out = self.sequential_extractor(out)        
        return out

## 3D CNN
    
class CNN_3D(TrainingPhase):
    def __init__(
            self,

    ):
        super().__init__()

    def forward(
            self,
            x
    ):
        
class CNN_2D(TrainingPhase):
    def __init__(
            self,

    ):
        super().__init__()

    def forward(
            self,
            x
    ):
        

class MC3_18(nn.Module):
    def __init__(self, hidden_state_size, architecture, hidden_activations, dropout: float = 0.1):
        from torchvision.models.video import mc3_18, MC3_18_Weights
        super(MC3_18, self).__init__()
        # (B, F, H, W) -> (B,F,C,H,W)
        self.input_layer = 

        self.model = mc3_18(weights = MC3_18_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            DeepNeuralNetwork(512, hidden_state_size, architecture, hidden_activations),
            nn.Dropout(dropout, inplace = True)
            )
        self.initial_transform = tt.Compose([
            tt.ToTensor(),
            tt.Resize((1024, 1024))
        ])

        self.intermediate_transform = tt.Compose([
            MC3_18_Weights.KINETICS400_V1.transforms(antialias = True)
        ])
    def forward(self, x):
        x = self.input_layer(x)
        x = self.intermediate_transform(x)
        x = self.model(x)
        return x
    

class MVIT_V2_S(nn.Module):
    def __init__(self, hidden_state_size, architecture, hidden_activations, dropout: float = 0.1):
        from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
        super(MVIT_V2_S, self).__init__()

        self.input_layer = 
        
        self.model = mvit_v2_s(weights = MViT_V2_S_Weights.KINETICS400_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Sequential(
            DeepNeuralNetwork(768, hidden_state_size, architecture, hidden_activations),
            nn.Dropout(dropout, inplace = True)
            )
        self.initial_transform = tt.Compose([
            tt.ToTensor(),
            tt.Resize((1024, 1024))
        ])

        self.intermediate_transform = tt.Compose([MViT_V2_S_Weights.KINETICS400_V1.transforms(antialias=True)])
    def forward(self, x):
        x = self.input_layer(x)
        x = self.intermediate_transform(x)
        x = self.model(x)
        return x

class SWIN3D_B(nn.Module):
    def __init__(self, hidden_state_size: int, architecture: tuple, hidden_activations: tuple, dropout: float = 0.1):
        from torchvision.models.video import swin3d_b, Swin3D_B_Weights
        super(SWIN3D_B, self).__init__()

        self.input_layer = 
        
        self.model = swin3d_b(weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.head = nn.Sequential(
            DeepNeuralNetwork(1024, hidden_state_size, architecture, hidden_activations),
            nn.Dropout(dropout, inplace = True)
            )
        self.initial_transform = tt.Compose([
            tt.ToTensor(),
            tt.Resize((1024, 1024))
        ])
        self.intermediate_transform = tt.Compose([Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms(antialias=True)])
    def forward(self, x):
        x = self.input_layer(x)
        x = self.intermediate_transform(x)
        x = self.model(x)
        return x

"""Vision transformer"""
"""
General utils for transformers
"""
class PatchEmbed_3DCNN(nn.Module):
    def __init__(
            self,
            d_model: int, 
            h_div: int,
            w_div: int,
            pe,
            feature_extractor,
            architecture: tuple, 
            hidden_activations: tuple, 
            X: torch.Tensor,
            dropout: float = 0.1,
    ):
        super().__init__()
        """
        B: batch size
        F: Frames
        C: Channels
        H_div: number of vertical cuts
        W_div: number of horizontal cuts
        H_div*W_div: total patches
        h: H/H_div
        w: W/W_div
        """
        assert (X.size(-2)%self.h_div ==0 or X.size(-1)%self.w_div == 0), 'Patching size must be multiplier of dimention'
        self.H_div = h_div
        self.W_div = w_div
        self.feature_extractor  = feature_extractor(d_model, architecture, hidden_activations, dropout)
        self.pe = pe
    def forward(
            self,
            X: torch.Tensor()
    ):
        B,F,C,H,W = X.shape
        # -> (B*h_div*w_div, F,C,h, w)
        X = X.view(-1,F,C, H//self.H_div, W//self.W_div)
        # -> (B,h_div*w_div, embed_size)
        out = self.feature_extractor(X).view(B, X.size(0)/B, -1)
        # -> (B,h_div*w_div, embed_size)
        X = self.pe(out)
        return X

class PatchEmbed_2DCNN(nn.Module):
    def __init__(
            self,
            d_model: int, 
            pe,
            feature_extractor,
            architecture: tuple, 
            hidden_activations: tuple, 
            dropout: float = 0.1,
    ):
        super().__init__()
        """
        B: batch size
        F: Frames
        C: Channels
        H_div: number of vertical cuts
        W_div: number of horizontal cuts
        H_div*W_div: total patches
        h: H/H_div
        w: W/W_div
        """
        self.d_model = d_model
        self.feature_extractor  = feature_extractor(d_model, architecture, hidden_activations, dropout)
        self.pe = pe
    def forward(
            self,
            X: torch.Tensor()
    ):
        B,F,C,H,W = X.shape
        # -> (B*F,C,H, W)
        X = X.view(B*F, C, H, W)
        # -> (B,F, embed_size)
        out = self.feature_extractor(X).view(B, F, -1)
        # -> (B,F, embed_size)
        X = self.pe(out)
        return X
        
    

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

def DnPositionalEncoding(
        x_n: torch.Tensor,
        delta_t: float,
        degree: int = 1,
):
    B,_,I = x_n.size()
    for i in range(1,degree+1):
        delta_x_n = torch.cat([torch.zeros(B,i,I), (x_n[i:]-x_n[:-i])/delta_t], dim = -2)
        x_n = x_n + delta_x_n
    return x_n

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
            seq_len,
            activation = nn.SiLU,
    ):
        super().__init__()
        self.activation = activation
        self.w_1 = nn.Linear(seq_len, seq_len)
        self.v = nn.Linear(seq_len, seq_len)
        self.w_2 = nn.Linear(seq_len, seq_len)
    def forward(
            self,
            x_n: torch.Tensor,
            k: int,
    ):
        super().__init__()
        x_n = x_n.tranpose(-1,-2)
        return self.w_2(self.activation(self.w_1(x_n)) + self.v(x_n)).transpose(-1,-2)


"""
VideoAnalysis Encoder

{PatchEmbedding (video-> patched feature 3D CNN wise/feature 2D CNN wise->positional encoding)
->
Spatio Temporal attention
->
Cross Attention
->
mlp} x N

out

"""

class VideoAnalysisEncoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout
    ):
        super().__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads),
            nn.MultiheadAttention(d_model, num_heads),
        ])
        self.mlp = PosWiseFFN(d_model)
        self.residual_connections = nn.ModuleList([PreNormResidual(RootMeanSquaredNormalization(d_model), dropout) for _ in range(3)])
    def self_attention(
            self,
            out
    ):
        return self.residual_connections[0](out, lambda x: self.attention[0](x,x,x))
    def cross_attention(
            self,
            self_out,
            cross_out
    ):
        """
        Q: cross_out
        K: self_out
        V: self_out
        """
        return self.residual_connections[1](self_out, lambda x: self.attention[1](cross_out, x,x))
    def poswise_mlp(
            self,
            out
    ):
        return self.residual_connections[2](out, lambda x: self.mlp(x))
"""
Timeseries Transfomer

{
embedding_layer,
rotary positional encoding -> Dn positional encoding,
time based self attention,
cross attention,
pos_wise_mlp + seq_wise_mlp,

}

"""
class TimeseriesEmbeddingAndPositionalEncoding(nn.Module):
    def __init__(
            self,
            input_size: int,
            d_model: int,
            seq_len: int,
            delta_t: float,
            degree: int

    ):
        if input_size != d_model:    
            self.embedding_layer = nn.Linear(input_size, d_model, False)
        self.rotary_pe = RotaryPositionalEncoding(d_model, seq_len)
        self.delta_t = delta_t
        self.degree = degree
    def forward(
            self,
            x
    ):
        if self.embedding_layer is not None:    
            x = self.embedding_layer(x)
        x = self.rotary_pe(x)
        x = DnPositionalEncoding(x, self.delta_t, self.degree)
        return x
    
class TimeseriesTransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            seq_len: int,
            num_heads: int,
            dropout: float
    ):
        super().__init__()
        self.attention = nn.ModuleList([
            ModedTimeSeriesAttention(d_model, num_heads),
            ModedTimeSeriesAttention(d_model, num_heads),
        ])
        self.seqwise = SeqWiseMLP(seq_len, activation = nn.GELU())
        self.poswise = PosWiseFFN(d_model, )
        self.residual_connections = nn.ModuleList([PreNormResidual(RootMeanSquaredNormalization(d_model), dropout) for _ in range(3)])
    def self_attention(
            self,
            x_n
    ):
        return self.residual_connections[0](x_n, lambda x: self.attention[0](x,x,x))
    def cross_attention(
            self,
            self_out,
            cross_out
    ):
        """
        Q: cross_out
        K: self_out
        V: self_out
        """
        return self.residual_connections[1](self_out, lambda x: self.attention[1](cross_out, x,x))
    def mlp(
            self,
            out
    ):
        return self.residual_connections[2](out, lambda x: self.poswise(x) + self.seqwise(x))    
    def forward(self, x_n):
        x_n = self.self_attention(x_n)
        x_n = self.mlp(x_n)
        return x_n
    
class TimeseriesTransformer(nn.Module):
    def __init__(
            self,
            input_size: int,
            d_model: int,
            seq_len: int,
            num_heads: int,
            num_layers: int,
            dropout: float,
            delta_t: float,
            degree: int,
            fc_args: dict
    ):
        super().__init__()
        self.embedding = TimeseriesEmbeddingAndPositionalEncoding(
            input_size, d_model, seq_len, delta_t, degree
        )

        self.model = nn.TransformerEncoder(
            TimeseriesTransformerEncoderBlock(d_model, seq_len, num_heads, dropout),
            num_layers
        )

        self.fc = DeepNeuralNetwork(d_model, fc_args['out_size'], fc_args['arch'], fc_args['act'], fc_args['out_act'])
    def feature_extraction(self, x_n):
        x_n = self.embedding(x_n)
        return self.model(x_n)
        
    def forward(self, x_n):
        x_n = self.feature_extraction(x_n)
        return self.fc(x_n)
    def forcing_forward(
            self,
            batch,
            criterion,
            noise_factor,
    ) -> tuple:
        forcing_loss = 0
        x_n_1 = torch.cat([batch['l1_ace'], batch['l1_dscovr']], dim = -1)
        x_n_2 = torch.cat([batch['l2_ace'], batch['l2_dscovr']], dim = -1)
        for _, encoder_layer in self.model.layers:
            x_n_1 = encoder_layer(x_n_1)
            x_n_2 = encoder_layer(x_n_2)
            forcing_loss += criterion(x_n_1, x_n_2, noise_factor)
        return x_n_1, forcing_loss
    
"""
Multimodal Block:
VideoAnalysisTransformer + TimeseriesTransformer
"""

class MultimodalTransformerEncoderBlock(nn.Module):
    def __init__(
            self,
            VideoBlock_args: dict,
            Timeseries_args: dict,
    ):
        super().__init__()
        self.sdo_encoder = VideoAnalysisEncoderBlock(VideoBlock_args.d_model, VideoBlock_args.num_heads,
                                              VideoBlock_args.dropout)
        self.l1_encoder = TimeseriesTransformerEncoderBlock(Timeseries_args.d_model, Timeseries_args.seq_len,
                                                            Timeseries_args.num_heads, Timeseries_args.dropout)
    def forward(
            self,
            X,
            x_n
    ):
        # Initial self attention layer
        X = self.sdo_encoder.self_attention(X)
        x_n = self.l1_encoder.self_attention(x_n)
        # Cross attention layer
        X_crossed = self.sdo_encoder.cross_attention(X,x_n)
        x_n_crossed = self.l1_encoder.cross_attention(x_n, X)
        #MLP
        X = self.sdo_encoder.poswise_mlp(X_crossed)
        x_n = self.l1_encoder.mlp(x_n_crossed)
        return X, x_n
    


"""Multimodal backbone"""

class MultimodalTransformer(nn.Module):
    def __init__(
            self, 
            SDO_args: dict,
            L1_args: dict,
            fc_args: dict,
            num_layers: int,
    ):
        super().__init__()
        # 1. Instantiate the video model parameters
        ## Patch embedding
        self.SDO_embedding = PatchEmbed_3DCNN(SDO_args.d_model, SDO_args.h_div, SDO_args.w_div, 
                                        RotaryPositionalEncoding(SDO_args.d_model, SDO_args.seq_len),
                                        SDO_args.feature_extractor,
                                        SDO_args.embed_arch['arch'],
                                        SDO_args.embed_arch['act'],
                                        SDO_args.dropout)
        self.L1_embedding = TimeseriesEmbeddingAndPositionalEncoding(L1_args.input_size, L1_args.d_model,
                                                                     L1_args.seq_len, L1_args.delta_t, L1_args.degree)
        ## Encoder backbone
        self.encoder = nn.TransformerEncoder(
            MultimodalTransformerEncoderBlock(SDO_args, L1_args),
            num_layers
        )
        ## Final Layer
        self.fc = DeepNeuralNetwork(L1_args.d_model + SDO_args.d_model, fc_args.output_size, fc_args.architecture,
                                    fc_args.hidden_activations, fc_args.out_activation)
    def feature_extraction(
            self,
            X, 
            x_n
    ):
        # Embedding, positional encoding, etc.
        x_n = self.L1_embedding(x_n)
        X = self.SDO_embedding(X)
        # Backbone process
        X, x_n = self.encoder(X, x_n)
        # Concat and forward
        return torch.cat([X, x_n], dim = -1)
    def forward(
            self,
            batch
    ):
        x_n = torch.cat([batch['l1_ace'], batch['l1_dscovr']], dim = -1)
        X = batch['sdo']
        out = self.feature_extraction(X, x_n)
        out = self.fc(out)
        return out
    def forcing_forward(
            self,
            batch,
            criterion,
            noise_factor,
    ) -> tuple:
        forcing_loss = 0
        x_n_1 = torch.cat([batch['l1_ace'], batch['l1_dscovr']], dim = -1)
        x_n_2 = torch.cat([batch['l2_ace'], batch['l2_dscovr']], dim = -1)
        X = batch['sdo']
        for _, encoder_layer in self.encoder.layer:
            X, x_n_1 = encoder_layer(X, x_n_1)
            X, x_n_2 = encoder_layer(X, x_n_2)
            forcing_loss += criterion(x_n_1, x_n_2, noise_factor)
        return torch.cat([X, x_n_1], dim = -1), forcing_loss

"""
Rnn Residual Encoder
"""
class RnnResidualEncoder(TrainingPhase):
    def __init__(
            self,
            d_model,
            rnn,
            dropout: float = 0.1,
            bidirectional: bool = False,
            num_layers: int = 4,
            output_size: int = 7,
            architecture: tuple = (), 
            hidden_activations: tuple = (),
            out_activation: nn = None
    ):
        super().__init__()
        assert (num_layers<6), 'Not aloud'
        self.rnn =nn.ModuleList([rnn(d_model, d_model, 1, dropout, bidirectional) for _ in range(num_layers)]) 
        self.residual_connections = nn.ModuleList([PreNormResidual(RootMeanSquaredNormalization(d_model), dropout)
                                                   for _ in range(num_layers)])
        self.fc = DeepNeuralNetwork(d_model, output_size, architecture, hidden_activations, out_activation)
    def feature_extraction(
            self,
            x_n
    ):
        for rnn, residual_connection in zip(self.rnn, self.residual_connections):
            x_n = residual_connection(x_n, lambda x: rnn(x))
        return x_n
    def forward(
            self,
            batch
    ):
        x_n = torch.cat([batch['ace'], batch['l1']], dim = -1)
        x_n = self.feature_extraction(x_n)
        x_n = self.fc(x_n)
        return x_n
    def forcing_forward(
            self,
            batch,
            criterion,
            noise_factor,
    ) -> tuple:
        forcing_loss = 0
        x_n_1 = torch.cat([batch['l1_ace'], batch['l1_dscovr']], dim = -1)
        x_n_2 = torch.cat([batch['l2_ace'], batch['l2_dscovr']], dim = -1)
        for rnn, residual_connection in zip(self.rnn, self.residual_connections):
            x_n_1 = residual_connection(x_n_1, lambda x: rnn(x))
            x_n_2 = residual_connection(x_n_2, lambda x: rnn(x))
            forcing_loss += criterion(x_n_1, x_n_2, noise_factor)
        return x_n_1, forcing_loss

        
"""
RNN based multimodal model
image_extractor: MC3_18 3D CNN model
l1_extractor: residual rnn model
"""
class MultimodalRnnModel(nn.Module):
    def __init__(
            self,
            rnn_args,
            conv_args,
            fc_args,
    ):
        super().__init__()
        self.l1_encoder = RnnResidualEncoder(rnn_args.d_model, rnn_args.rnn, rnn_args.dropout, rnn_args.bidirectional,
                                             rnn_args.num_layers)
        self.sdo_encoder = MC3_18(conv_args.hidden_state_size, conv_args.architecture, conv_args.hidden_activations,
                                  conv_args.dropout)
        self.fc = DeepNeuralNetwork(rnn_args.d_model + conv_args.hidden_state_size, 
                                    fc_args.output_size, fc_args.architecture, fc_args.hidden_activations, fc_args.out_activation)

    def feature_extraction(
            self,
            X,
            x_n
    ):
        x_n = self.l1_encoder.feature_extraction(x_n)
        X = self.sdo_encoder(X)
        return torch.cat([X, x_n], dim = -1)
    def forward(
            self,
            batch
    ):
        return self.fc(self.feature_extraction(batch['sdo'],batch['l1']))
    def forcing_forward(
            self,
            batch,
            forcing_criterion,
            noise_factor
    ) -> tuple:
        x_n, forcing_loss = self.l1_encoder.forcing_forward(batch, forcing_criterion, noise_factor)
        X = self.sdo_encoder(batch['sdo'])
        return torch.cat([X, x_n], axis = -1), forcing_loss
    