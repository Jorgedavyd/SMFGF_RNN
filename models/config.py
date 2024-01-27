from models.utils import get_default_device
import torch
import os

def create_rnn_config():
  return {
    'name': 'RnnBased_LSTM',
    'device': get_default_device(),
    'last_epoch': None,
    'global_step_train': 0,
    'global_step_val': 0,
    'optimizer_state_dict': None,
    'scheduler_state_dict': None,
  }
  
def last_config():
  try:  
    files = os.listdir('./models/models')
    for idx, file in enumerate(files):
      print(f'{idx+1}. {file}')
    config = int(input('Choose the last config: '))
    return torch.load(files[config-1])
  except FileNotFoundError:
    return create_rnn_config()
  