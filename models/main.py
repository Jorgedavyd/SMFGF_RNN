from models.utils import RegressionTrainingPhase
import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import datetime, timedelta

## RMSN
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

## Attention
    
class FlashAttention(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
    @staticmethod
    def attention(queries, keys,values, mask = None, dropout = 0):
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
                dropout
            )
        return out
    
## LSTM
    
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

## GRU

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


class GeomagneticRNN:
  pretraining = True
  DSCOVR_size: int = 7 #DSCOVR vector size (AUMENTAR CON feature engineering)
  h: int = 100 #Hyperparameter
  SWARM_size: int = 8 #SWARM vector size (AUMENTAR CON feature engineering)
  target_size: int = SWARM_size if pretraining else 1
  #RNN 
  S: timedelta = timedelta(days = 2) #Sequence length (2 dias)
  N: int  = 5 # Num RNN layers
  bidirectional: bool = False
  dropout: int = 0. 
  attention = {
    'cross': [h,h,h, 10],
    'masked': [target_size, h, h, 3]
  }
  rnn = {
    'encoder': [DSCOVR_size, h, N, dropout, bidirectional],
    'decoder': [h, h, N, dropout, bidirectional]
  }
  rnn_type = LSTM
  
class RnnEncoder(nn.Module):
  def __init__(
      self,
      args: GeomagneticRNN
  ):
    super().__init__()
    # Encoder
    ## Layer norm
    self.rmsn_encoder = RootMeanSquaredNormalization(args.DSCOVR_size)
    ## RNN
    self.encoder = args.rnn_type(*args.rnn['encoder'])
  def forward(self, x_n):
    ## Root mean squared norm
    x_n = self.rmsn_encoder(x_n)
    ## (B,S,input_size) -> (B,S,num_layers, hidden_size)
    x_n = self.encoder(x_n)
    return x_n

class RnnDecoder(nn.Module):
  def __init__(
      self,
      args: GeomagneticRNN
  ):
    super().__init__()
    #Masked attention
    self.attention = nn.ModuleList([
      ModedTimeSeriesAttention(*args.attention['mask']),
      ModedTimeSeriesAttention(*args.attention['cross'])
    ])
    ## ResLayer
    self.ResLayer = nn.ModuleList([PreNormResidual(RootMeanSquaredNormalization(args.target_size), args.dropout)] + \
    [PreNormResidual(RootMeanSquaredNormalization(args.h), args.dropout) for _ in range(3)])   
    #RNN
    self.rnn = args.rnn_type(*args.rnn['decoder'])
    #Layer norm
    self.rmsn= RootMeanSquaredNormalization(args.target_size),
    #SwiGLU
    self.activation = nn.SiLU()
    self.w_1 = nn.Linear(args.h, args.h)
    self.v = nn.Linear(args.h, args.h)
    self.w_2 = nn.Linear(args.h, args.h)
  def forward(self, x_n, y_n, x_mask = None, y_mask = None):
    out = self.ResLayer[0](y_n, lambda x: self.attention[0](x, x,x, y_mask))
    out = self.ResLayer[1](out, lambda x: self.rnn(x))
    out = self.ResLayer[2](out, lambda x: self.attention[1](x, x_n, x_n, x_mask))
    out = self.ResLayer[3](out, lambda x: self.w_2(self.activation(self.w_1(x)) + self.v(x)))
    return out

class Model(RegressionTrainingPhase):
  def __init__(self, encoder: RnnEncoder, decoder: RnnDecoder, args: GeomagneticRNN):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder 
    self.fc = nn.Linear(args.h, args.SWARM_size)
    self.build_model()
  def projection(self, x_n):
    return self.fc(x_n)    
  def build_model(self):
    for parameter in self.parameters():
      if parameter.dim > 1:
        nn.init.xavier_uniform_(parameter)
  def forward(self, x, y, x_mask = None, y_mask = None):
    x = self.encoder(x)
    x = self.decoder(x, y, x_mask, y_mask)
    return self.projection(x)

class ModedTimeSeriesAttention(nn.Module):
  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_heads: int
  ):
    super().__init__()
    """
    attention = (concat({head_i}_{i=1}^{num_heads})W_fc)^T
    head_i = softmax(\frac{Q_{i}^{T}K}{\sqrt(d_{k})})V^{T}
    """
    # Premises
    assert (hidden_dim%num_heads == 0), 'k = \frac{hiden_dim}{num_heads} | k \in \mathbb(Z)^{+}'
    self.num_heads = num_heads
    self.head_dim = hidden_dim/num_heads
    # Projections into hidden
    self.W_q = nn.Linear(input_dim, hidden_dim)
    self.W_k = nn.Linear(input_dim, hidden_dim)
    self.W_v = nn.Linear(input_dim, hidden_dim)
    # fc projection
    self.W_fc = nn.Linear(self.head_dim*num_heads, output_dim)
  def forward(self, queries, keys, values, mask = None):
    #Dimensions
    B,S,_ = queries.shape
    #Linear projection
    Q = self.W_q(queries)
    K = self.W_k(keys)
    V = self.W_v(values)
    # Q = Q^T, K = K^T, V = V^T so that attention = soft(Q^T K / sqrt(d_k))V^T -> (B, num_heads, head_dim, seq_len)
    Q_time = Q.view(B,S,self.num_heads, self.head_dim).transpose(1,2).transpose(-1,-2)
    K_time = K.view(B,S,self.num_heads, self.head_dim).transpose(1,2).transpose(-1,-2)
    V_time = V.view(B,S,self.num_heads, self.head_dim).transpose(1,2).transpose(-1,-2)

    # (B,num_heads, head_dim, S)-> (B,num_heads, S, head_dim) -> (B,S,num_heads, head_dim)
    out_time = FlashAttention.attention(Q_time,K_time,V_time).transpose(-1,-2).transpose(-2,1) 
    out_time = out_time.view(B,S,self.num_heads*self.head_dim) # -> (B,S,hidden_dim)

    # (B,S,num_heads, head_dim) -> (B,S,hidden_dim)
    stan_out = FlashAttention.attention(Q,K,V, mask).view(B,S,self.num_heads*self.head_dim)

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
