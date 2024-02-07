from models.main import *
from ftplib import FTP
import xgboost as xgb
import pandas as pd
import requests
import joblib

class MainPipeline(nn.Module):
  def __init__(
      self,
      model_arch,
      model_args,
      backbone_location,
      rf_location,
      boosting_location,
      scaler_location
  ):
    super().__init__()
    self.input_scaler = joblib.load(scaler_location)
    #cambiar
    self.backbone = model_arch(*model_args)
    self.backbone.load_state_dict(torch.load(backbone_location)['model_state_dict'])
    self.rf = joblib.load(rf_location)
    self.boosting = xgb.Booster().load_model(boosting_location)
  def import_dat(self):
    joint_col_names = []
    #DSCOVR
    fc_url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    mag_url = 'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
    fc_response = requests.get(fc_url) 
    if fc_response.status_code == 200:
      fc_data = fc_response.json()
    fc = pd.DataFrame(fc_data[1:], columns = fc_data[0]).set_index('time_tag', drop = True)
    fc.index = pd.to_datetime(fc.index)
    mg_response = requests.get(mag_url) 
    if mg_response.status_code == 200:
      mg_data = mg_response.json()
    mg = pd.DataFrame(mg_data[1:], columns = mg_data[0]).set_index('time_tag', drop = True)
    mg.index = pd.to_datetime(mg.index)
    dscovr = pd.concat([fc, mg], axis = 1)
    #ACE
    datasets_column_names = {
      'swepam': ,
      'epam': ,
      'mag': ,
      'sis': ,
    }
    datasets = {

    }
    ace_prep = {
      'swepam': ,
      'epam': ,
      'mag': ,
      'sis': ,
    }
    ace_line_idx = {
      'swepam': 18,
      'epam': 18,
      'mag': 20,
      'sis': 16,
    }
    with FTP('ftp.swpc.noaa.gov') as ftp:
      ftp.login()
      files = [
        './pub/lists/ace/ace_swepam_1m.txt',
        './pub/lists/ace/ace_sis_1m.txt',
        './pub/lists/ace/ace_mag_1m.txt',
        './pub/lists/ace/ace_epam_1m.txt'
      ]

      for file_to_download in files:
        with open(file_to_download.split('/')[-1], 'wb') as local_file:
          ftp.retrbinary(f'RETR {file_to_download}', local_file.write)
          data = []
          for line in local_file.readlines()[ace_line_idx[key]:]:
            data.append(line.join())
          datasets[file_to_download.split('_')[-2]] = data
      ftp.quit()

    for key, value in datasets:
      datasets[key] = pd.DataFrame(value, columns = datasets_column_names[key])
      ace_prep[key](datasets[key])
    ace = pd.concat(datasets.values(), axis = 1)
    
    data = pd.concat([ace, dscovr], axis = 1)
    data.columns = joint_col_names
    
    # Feature engineering


    return torch.from_numpy(self.input_scaler.transform(data.values)).to(torch.float16)

  def pipeline(self):
    x = self.import_data()
    out = self.backbone.extract_features(x)

    input_rf = out.detach().numpy()
    input_boosting = xgb.DMatrix(input_rf)
    rf_output = self.rf.predict(input_rf)
    boosting_output = self.boosting.predict(input_boosting)
    linear_output = self.backbone.fc(out)

    return rf_output, boosting_output, linear_output

