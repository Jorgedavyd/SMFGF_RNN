from sklearn.preprocessing import StandardScaler
from models.utils import map_to_discrete
from torch.utils.data import Dataset
from data.preprocessing import *
from datetime import datetime
import torch.nn as nn
import torch

#Interesting intervals to scrap from

class DscovrScrapIntervals:
    less_for = [
        ('2023-04-18', '2023-04-27'),
    ]
    
    five = [
        ('2023-04-18', '2023-04-27'),
    ]
    
    six_seven = [
        ('2023-04-18', '2023-04-27'),
        ('2023-04-18', '2023-04-27'),
        
    ]
    
    eight_nine = [
        ('2023-04-18', '2023-04-27'),
        ('2023-03-19', '2023-03-26'),
        ('2021-10-30', '2021-11-06'),
        ('2017-09-03', '2017-09-11'),
    ]
    
    scrap_date_list = less_for + five + six_seven + eight_nine

class SyntheticDataCreation:
    
    dates_interest = [
        ('2015-06-18', '2015-06-25'),
        ('2015-03-13', '2015-03-19'),
        ('2013-09-30', '2013-10-04'),
        ('2012-03-05', '2012-03-11'),
        ('2011-08-01', '2011-03-07'),
        ('2010-04-01', '2010-04-07'),
        ('2006-12-11', '2006-12-17'),
        ('2005-08-20', '2005-08-26'),
        ('2005-05-04', '2005-05-17'),
        ('2005-01-15', '2005-01-23'),
        ('2005-09-07', '2005-09-13'),
        ('2005-05-26', '2005-10-02'),
        ('2005-05-26', '2005-10-02'),
        ('2005-01-03', '2005-01-09'),
        ('2004-07-21', '2004-07-29'),
        ('2004-11-03', '2004-07-12'),
        ('2003-10-25', '2003-11-03'),
        ('2003-05-25', '2003-05-02'),
        ('2003-11-16', '2003-11-22'),
        ('2001-03-27', '2001-04-02'),
        ('2001-11-02', '2001-11-08'),
        ('2001-11-20', '2001-11-26'),
        ('2001-04-07', '2001-04-13'),
        ('2001-10-17', '2001-10-23'),
    ]


class PretrainingDefault(Dataset):
    def __init__(
        self, 
        dscovr: list, 
        dscovr_sl: timedelta, 
        swarm: pd.DataFrame,
        dst: pd.DataFrame, 
        step_size:timedelta = timedelta(minutes = 5), 
        swarm_dst_sl = timedelta(hours = 3), 
        deviation = timedelta(minutes = 20)
    )-> tuple:
        self.dscovr_sl = time_to_step(dscovr_sl, step_size)
        self.dscovr= dscovr
        self.swarm = swarm
        self.dst = dst
        self.output_deviation = time_to_step(deviation, step_size)
        self.swarm_dst_sl = time_to_step(swarm_dst_sl, step_size)
    def __len__(self):
        return self.dscovr[0].shape[0] - self.dscovr_sl + 1
    def __getitem__(self, idx):
        dscovr = [dscovr.values[idx: idx+self.dscovr_sl, :] for dscovr in self.dscovr]
        output_init = idx+self.dscovr_sl+self.output_deviation
        swarm = self.swarm.values[output_init: output_init + self.swarm_dst_sl]
        dst = self.dst.values.reshape(-1,1)[output_init:output_init+self.swarm_dst_sl]
        return dscovr, swarm, dst

class Swarm2Dst(Dataset):
    def __init__(self, input_seqs: list, output_seq, seq_len, step_size = timedelta(minutes = 5)):
        self.seq_len = time_to_step(seq_len, step_size)
        self.input_seqs= input_seqs
        self.output_seqs = output_seq
    def __len__(self):
        return self.input_seqs[0].shape[0] - self.seq_len + 1
    def __getitem__(self, idx):
        input_seqs = [input_seq.values[idx: idx+self.seq_len, :] for input_seq in self.input_seqs]
        output_seq = self.output_seqs.values.reshape(-1,1)[idx:idx+self.seq_len, :]
        
        return input_seqs, output_seq

def time_shift(scrap_date, dev, sl):
    init_date = datetime.strptime(scrap_date[0],'%Y-%m-%d')
    end_date = datetime.strptime(scrap_date[-1],'%Y-%m-%d') + timedelta(days = 1) + dev + sl
    return [init_date, end_date]

def time_to_step(time, step_size=timedelta(minutes=5)):
    return int(time.total_seconds() / step_size.total_seconds())

def dst_time_shift(scrap_date, dev, sl):
    init_date = datetime.strptime(scrap_date[0], '%Y-%m-%d')
    end_date = datetime.strptime(scrap_date[-1], '%Y-%m-%d') + timedelta(days = 1) + dev + sl
    init_date = datetime.strftime(init_date, '%Y%m%d')
    end_date = datetime.strftime(end_date, '%Y%m%d')
    scrap_date = interval_time(init_date,end_date, format = '%Y%m%d')
    return scrap_date

class GeneralDataset(Dataset):
    def __init__(
            self, 
            scrap_date_list: list,
            swarm_sc: str = 'A',
            swarm_dst_sl: timedelta = timedelta(hours = 3),
            dscovr_sl: timedelta = timedelta(days = 2),
            step_size: timedelta = timedelta(minutes = 5),
            deviation: timedelta = timedelta(minutes = 20)
    ):
        """
        Pretraining dataset

        output:
        
        DSCOVR L1 & L2 data
        SWARM_X spacecraft
        Dst index

        """
        self.name_list = ['l1', 'l2', 'swarm', 'dst']
        self.dscovr_sl = dscovr_sl
        self.swarm_dst_sl = swarm_dst_sl
        # Defining spacecrafts
        dscovr = DSCOVR()
        swarm = SWARM()
        ## scrap_date list
        l1_scrap = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        dscovr_temp = []
        swarm_temp = []
        self.datasets = [] # save on Sequential Dataset method
        for scrap_date in l1_scrap:
                
                #DSCOVR
                dscovr_seqs = [
                    tool.resample(step_size).mean() for tool in dscovr.MAGFC(scrap_date, 'both', True)
                ] #creating the list of satellite data
                
                ##SWARM
                swarm_scrap = time_shift(scrap_date, deviation, swarm_dst_sl)
                swarm_data = (swarm.MAG_x(swarm_scrap, swarm_sc).resample(step_size).mean().drop(['Longitude', 'Dst', 'Radius', 'Latitude'], axis = 1))
                
                ## DST INDEX
                dst_scrap = dst_time_shift(scrap_date, deviation, swarm_dst_sl)
                dst = Dst(dst_scrap)
                
                self.datasets.append(PretrainingDefault(dscovr_seqs, dscovr_sl, swarm_data, dst, step_size, swarm_dst_sl, deviation))

                dscovr_temp.append(dscovr_seqs)
                swarm_temp.append(swarm_data)
                
        self.dscovr_scalers = []
        for i in range(len(dscovr_seqs)):
            temp_input = [input_seqs[i] for input_seqs in dscovr_temp]
            self.dscovr_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

        self.swarm_scaler = StandardScaler().fit(pd.concat(swarm_temp, axis = 0).values)
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        
        dscovr, swarm, dst = self.datasets[dataset_idx][index]
        
        transformed_dscovr = []
        for scaler, data in zip(self.dscovr_scalers, dscovr):
            transformed_dscovr.append(torch.from_numpy(scaler.transform(data)).to(torch.float16))

        del dscovr

        swarm = torch.from_numpy(self.swarm_scaler.transform(swarm)).to(torch.float16)

        dst = map_to_discrete(dst)

        return dict(zip(self.name_list, [*transformed_dscovr, swarm, dst]))

