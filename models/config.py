from models.utils import get_default_device
import torch
import os

def create_config(
    name_run: str
):
  return {
    'name': name_run,
    'device': get_default_device(),
    'last_epoch': None,
    'global_step_train': 0,
    'global_step_val': 0,
    'optimizer_state_dict': None,
    'scheduler_state_dict': None,
  }
  
def import_config():
  try:  
    files = os.listdir('./models/models')
    for idx, file in enumerate(files):
      print(f'{idx+1}. {file}')
    config = int(input('Choose the config: '))
    return torch.load(files[config-1])
  except FileNotFoundError:
    return create_config()
  