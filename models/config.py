from models.utils import get_default_device

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
  
  