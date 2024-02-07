from sklearn.preprocessing import StandardScaler
from models.utils import map_to_discrete
from torch.utils.data import Dataset
from torchvision import transforms
from data.preprocessing import *
from datetime import datetime
from PIL import Image
import torch
def classify_dates(dates):

    dates = [datetime.strptime(date[0],"%Y/%m/%d") for date in dates]
    dates.sort()
    classified_dates = []
    current_interval = [dates[0]]
    for i in range(1, len(dates)):
        delta = (dates[i] - timedelta(days = 4)) - (dates[i-1] + timedelta(days = 2))
        if delta <= timedelta(days=2):
            current_interval.append(dates[i])
        else:
            classified_dates.append(current_interval)
            current_interval = [dates[i]]
    
    classified_dates.append(current_interval)

    scrap_interval = lambda interval: (datetime.strftime((interval[0] - timedelta(days = 4)), "%Y-%m-%d")\
                                   , datetime.strftime((interval[-1] + timedelta(days = 2)), "%Y-%m-%d"))
    
    scrap_full_date = [scrap_interval(interval) for interval in classified_dates]

    return scrap_full_date

#Interesting date intervals
class SdoScrapIntervals:
    dates = []
    with open('data/mag_storms_sdo.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            dates.append((line.split()[1], max([int(i) for i in line.replace('-', '').replace('+','').split()[3:]])))
    
    scrap_date_list = classify_dates(dates)

class DscovrScrapIntervals:
    dates = []
    with open('data/mag_storms_dscovr.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            dates.append((line.split()[1], max([int(i) for i in line.replace('-', '').replace('+','').split()[3:]])))
    
    scrap_date_list = classify_dates(dates)
    
    full_date_list = [('2016-07-30', '2024-01-24')]

    
class SyntheticDataCreation:
    dates = []
    with open('data/mag_storms_synth.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            dates.append((line.split()[1], max([int(i) for i in line.replace('-', '').replace('+','').split()[3:]])))
    
    scrap_date_list = classify_dates(dates)
    
    

"""
Pretraining Default

Default dataset architecture for pretraining purposes.

Output:
DSCOVR (L1,L2)
SWARM_X
Dst

"""
class PretrainingDefault(Dataset):
    def __init__(
        self, 
        input : list, 
        input_sl: timedelta, 
        target_sl: timedelta, 
        swarm: pd.DataFrame,
        dst: pd.DataFrame, 
        step_size:timedelta = timedelta(minutes = 5), 
    )-> tuple:
        self.input_sl = time_to_step(input_sl, step_size)
        self.input = input
        self.swarm = swarm
        self.dst = dst
        self.target_sl = time_to_step(target_sl, step_size)
    def __len__(self):
        return self.input[0].shape[0] - self.input_sl + 1
    def __getitem__(self, idx):
        input_sequence = [input_seq[idx: idx+self.input_sl, :] for input_seq in self.input]
        swarm = self.swarm[idx: idx+self.target_sl, :]
        dst = self.dst[idx: idx+self.target_sl, :]
        return input_sequence, swarm, dst

"""
Default Reconstruction dataset
output:
L1 corrupted
L1 not corrupted
"""    
class ReconstructionDefault(Dataset):
    def __init__(
            self,
            input_seq: list,
            sequence_length: timedelta,
            step_size: timedelta
    ):
        self.sequence_length = time_to_step(sequence_length, step_size)
        self.input_sequence = input_seq
    def __len__(self):
        return self.input_sequence.shape[0] - self.sequence_length + 1
    def __getitem__(self, idx):
        return self.input_sequence[idx: idx+self.sequence_length, :]
        
"""
Default Image dataset

output:
SDO images
Dst index
"""
class SecuenceImageDstDataset(Dataset):
    def __init__(
            self,
            names: list,
            dst,
            image_seq_len,
            dst_seq_len
    ):
        super().__init__()
        self.file_loc = lambda x: './data/SDO/Joint' + x
        self.dst = dst
        self.image_seq_len = image_seq_len
        self.dst_seq_len = dst_seq_len
        self.names = [self.file_loc(name) for name in names]
        self.images = [plt.imread(fname) for fname in self.names]

    def __len__(self):
        return len(self.images) - self.image_seq_len + 1
    def __getitem__(self, idx):
        output = {}
        #Dst index
        output['target'] = self.dst[idx: idx+self.dst_seq_len, :]
        #Imagenes
        output['input'] = self.images[idx: idx+self.image_seq_len, :]

        return output

"""
Default dataset architecture for supervised synthetic data.

Output:
input_seqs: [DSCOVR_L1, DSCOVR_L2, *other satellites]
dst: Dst index
"""
class SyntheticDefault(Dataset):
    def __init__(
        self, 
        input_seqs: list, 
        target_seq: list,
        dst: pd.DataFrame, 
        seq_len: timedelta, 
        step_size:timedelta = timedelta(minutes = 5), 
    )-> tuple:
        self.seq_len = time_to_step(seq_len, step_size)
        self.input_seqs= input_seqs
        self.dst = dst
        self.output_deviation = time_to_step(deviation, step_size)
        self.dst_sl = time_to_step(dst_sl, step_size)
    def __len__(self):
        return self.input_seqs[0].shape[0] - self.seq_len + 1
    def __getitem__(self, idx):
        input_seqs = [input_seq.values[idx: idx+self.seq_len, :] for input_seq in self.input_seqs]
        output_init = idx+self.seq_len+self.output_deviation
        dst = self.dst.values.reshape(-1,1)[output_init:output_init+self.dst_sl]
        return input_seqs, dst
"""
Default Second stage modeling
SWARM -> Dst
output: 
SWARM
Dst
"""
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

def time_shift(scrap_date, input_seq_len: timedelta, target_seq_len: timedelta, eps = timedelta(seconds = 1)):
    init_date = datetime.strptime(scrap_date[0],'%Y-%m-%d') + input_seq_len
    end_date = datetime.strptime(scrap_date[-1],'%Y-%m-%d') + target_seq_len + eps
    return [init_date, end_date]    

def time_to_step(time, step_size=timedelta(minutes=5)):
    return int(time.total_seconds() / step_size.total_seconds())

def dst_time_shift(scrap_date, input_seq_len: timedelta, target_seq_len: timedelta, eps = timedelta(seconds = 1)):
    init_date = datetime.strptime(scrap_date[0], '%Y-%m-%d') + input_seq_len
    end_date = datetime.strptime(scrap_date[-1], '%Y-%m-%d') + eps + target_seq_len
    init_date = datetime.strftime(init_date, '%Y%m%d')
    end_date = datetime.strftime(end_date, '%Y%m%d')
    scrap_date = interval_time(init_date,end_date, format = '%Y%m%d')
    return scrap_date

"""
Joint Dataset
DSCOVR + ACE + SOHO -> SWARM -> Dst

"""
class L1_to_SWARM(Dataset):
    def __init__(
            self, 
            scrap_date_list: list,
            swarm_sc: str = 'A',
            target_sl: timedelta = timedelta(hours = 3),
            input_sl: timedelta = timedelta(days = 2),
            step_size: timedelta = timedelta(minutes = 5),
    ):
        """
        Pretraining dataset

        output:
        
        SDO data (add)
        DSCOVR L1 & L2 data
        SWARM_X spacecraft
        Dst index

        """
        self.name_list = ['l1', 'l2', 'swarm', 'dst']
        self.input_sl = input_sl
        self.target_sl = target_sl
        # Defining spacecrafts
        dscovr = DSCOVR()
        ace = ACE()
        swarm = SWARM()
        ## scrap_date list
        l1_scrap = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        input_seqs_temp = []
        swarm_temp = []
        self.datasets = [] # save on Sequential Dataset method
        for scrap_date in l1_scrap:
                
                #DSCOVR & ACE 
                input_seqs = [
                    *[tool.resample(step_size).mean() for tool in dscovr.MAGFC(scrap_date, 'both', True)],
                    ace.MAG(scrap_date),
                    ace.SWEPAM(scrap_date),
                    ace.EPAM(scrap_date),
                    ace.SIS(scrap_date)
                ] #creating the list of satellite data
                
                ##SWARM
                swarm_scrap = time_shift(scrap_date, self.input_sl, self.target_sl)
                swarm_data = (swarm.MAG_x(swarm_scrap, swarm_sc).resample(step_size).mean().drop(['Longitude', 'Dst', 'Radius', 'Latitude'], axis = 1))
                
                ## DST INDEX
                dst_scrap = dst_time_shift(scrap_date, self.input_sl, self.target_sl)
                dst = Dst(dst_scrap)
                
                self.datasets.append(PretrainingDefault(input_seqs, input_sl, swarm_data, dst,\
                                                        step_size, target_sl))

                input_seqs_temp.append(input_seqs)
                swarm_temp.append(swarm_data)
                
        self.input_scalers = []
        for i in range(len(input_seqs)):
            temp_input = [input_seqs[i] for input_seqs in input_seqs_temp]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

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

"""
Second Stage Modeling

SWARM -> Dst
"""

class SecondStageModeling(Dataset):
    def __init__(self, scrap_date_list, seq_len = timedelta(hours = 2), step_size = timedelta(minutes = 5), dev = timedelta(minutes = 20)):
        """
        SWARM to DST

        This method targets DST sequences, we have these parameters:
        
        scrap_date_list: List of date intervals where you want to train your model. #lets to better dataset homogeneization
        seq_len: Sequence length.

        swarm: SWARM satellite object
        resample_method: Time step size
        datasets: List of build-in SequentialDataset objects

        resample(resample_method).mean() is set in order to organize the data on identical
        time intervals
        
        """
        self.name_list = ['swarm_a', 'swarm_b', 'swarm_c', 'dst']
        swarm = SWARM()
        scrap_list = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        inputs_temp = []
        output_temp = []
        self.datasets = [] # save on Sequential Dataset method
        swarm_sc =[
            'A',
            'B',
            'C'
        ]
        for scrap_date in scrap_list:
                swarm_scrap = time_shift(scrap_date, dev, seq_len)
                input_seqs = [swarm.MAG_x(swarm_scrap, letter).resample(step_size).mean().drop(['Longitude', 'Dst', 'Radius', 'Latitude'], axis = 1) for letter in swarm_sc]
                
                dst_scrap = dst_time_shift(scrap_date, dev, seq_len)
                output = Dst(dst_scrap)

                inputs_temp.append(input_seqs)
                output_temp.append(output)

                self.datasets.append(Swarm2Dst(input_seqs, output, seq_len, step_size))
            
        self.input_scalers = []
        for i in range(len(input_seqs)):
            temp_input = [input_seqs[i] for input_seqs in inputs_temp]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

        self.output_scaler = StandardScaler().fit(pd.concat(output_temp, axis = 0).values.reshape(-1,1))
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
        
        input_seqs, output = self.datasets[dataset_idx][index]
        
        transformed_inputs = []
        for scaler, data in zip(self.input_scalers, input_seqs):
            transformed_inputs.append(torch.from_numpy(scaler.transform(data)).to(torch.float32))

        del input_seqs

        transformed_output = torch.from_numpy(self.output_scaler.transform(output)).to(torch.float32)

        del output

        output = [*transformed_inputs, transformed_output]
        
        return dict(zip(self.name_list, output))
"""
Synthetic Dataset

WIND + ACE + SOHO -> DSCOVR

weighted towards 5,6,7,8,9 kp index

"""
class SyntheticDataset(Dataset):
    def __init__(
            self,
            scrap_date_list: list,
            step_size: timedelta,
            sequence_length: timedelta,
            dst_sl: timedelta,
            deviation: timedelta,

    ):
        """
        Data synthetic creation
        WIND,SOHO,ACE ----> DSCOVR (MG,FC)
        """
        self.name_list = ['target', 'ace_epam', 'ace_mag', 'soho_pm', 'soho_sem', 'wind_mag',  'dst']
        self.seq_len = sequence_length
        dscovr = DSCOVR()
        ace = ACE()
        soho = SOHO()
        wind = WIND()
        ##DSCOVR scrap_date list
        l1_scrap = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        inputs_temp = []
        output_temp = []
        self.datasets = [] # save on Sequential Dataset method
        for scrap_date in l1_scrap:
                
                input_seqs = [
                    *[tool.resample(step_size).mean() for tool in dscovr.MAGFC(scrap_date, joint = True)],
                    ace.SWEPAM(scrap_date),
                    ace.MAG(scrap_date),
                    # ace.SWEPAM(scrap_date),
                    # ace.SIS(scrap_date),
                    soho.CELIAS_PM(scrap_date),
                    soho.CELIAS_SEM(scrap_date),
                    # soho.COSTEP_EPHIN(scrap_date),
                    wind.MAG(scrap_date),
                    # wind.SMS(scrap_date),
                    # wind.TDP_PLSP(scrap_date),
                    # wind.TDP_PM(scrap_date),
                    # wind.SWE_electron_moments(scrap_date),
                    # wind.SWE_alpha_proton(scrap_date),
                    
                ] #creating the list of satellite data
                
                dst_scrap = dst_time_shift(scrap_date, deviation, dst_sl)
                dst = Dst(dst_scrap)

                self.datasets.append(SyntheticDefault(input_seqs, sequence_length, dst, step_size, dst_sl, deviation))

                inputs_temp.append(input_seqs)
                output_temp.append(dscovr)
        self.input_scalers = []
        for i in range(len(input_seqs)):
            temp_input = [input_seqs[i] for input_seqs in inputs_temp]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

        self.output_scaler = StandardScaler().fit(pd.concat(output_temp, axis = 0).values)
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
        
        input_seqs, dst = self.datasets[dataset_idx][index]
        
        transformed_inputs = []
        for scaler, data in zip(self.input_scalers, input_seqs):
            transformed_inputs.append(torch.from_numpy(scaler.transform(data)).to(torch.float16))

        del input_seqs

        return {
            'input': torch.cat(transformed_inputs[1:], dim = -1),
            'dscovr': transformed_inputs[0],
            'dst': dst.to(torch.float16)
        }
"""
SDO Reconstruction dataset

SDO corrupted -> SDO not corrupted
"""
class ReconstructionDatasetSDO(Dataset):
    def __init__(
            self,
            scrap_date_list: tuple,
            seq_len: timedelta,
            corrupted_transform: transforms.Compose,
            normal_transform: transforms.Compose
    ):
        sdo = SDO()
        self.names = []
        for scrap_date in scrap_date_list:
            self.names += sdo.AIA_HMI((scrap_date[0]-seq_len, scrap_date[-1]))

        self.corrupted_transform = corrupted_transform
        self.normal_transform = normal_transform
    def __len__(self):
        len(self.names)
    def __getitem__(self, idx):
        img = Image.open('./data/SDO/Joint/' + self.names[idx]).convert('RGB')
        corrupted_img = self.corrupted_transform(img)
        img = self.normal_transform(img)
        return {
            'input': corrupted_img,
            'target': img
        }
"""
Reconstruction transforms
"""

def RandomNoise(img, noise_factor: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(img) * noise_factor
    noisy_img = img + noise
    noisy_img = torch.clamp(noisy_img, 0, 1)
    return noisy_img
    
def RandomDropPixels(img, drop_probability: float=0.1):
    mask = torch.rand_like(img) > drop_probability
    masked_img = img * mask
    return masked_img

"""
Scaler Transorms
"""
class MinMaxScaler():
    def __init__(
            self,

    ):
        pass
    
class StandardScaler():
    def __init__(
            self,

    ):
        pass
"""
SDO Fine tuning dataset (on pretrain mode)

SDO only -> Dst


"""
    
class FineTuningImageDataset(Dataset):
    def __init__(
            self,
            scrap_date_list: list,
            image_seq_len: timedelta,
            dst_seq_len: timedelta = timedelta(days = 4),
            transform: transforms.Compose = transforms.Compose(transforms.ToTensor())
    ):
        sdo = SDO()
        self.datasets = []
        for scrap_date in scrap_date_list:
            names = sdo.AIA_HMI((datetime.strptime(scrap_date[0], "%Y-%m-%d")-image_seq_len, \
                                 datetime.strptime(scrap_date[-1],"%Y-%m-%d")))
            #Dst
            dst_scrap = dst_time_shift(scrap_date, timedelta(minutes = 0), dst_seq_len, True)
            dst = Dst(dst_scrap)

            self.datasets.append(SecuenceImageDstDataset(names, dst, image_seq_len, dst_seq_len))
            
        self.transform = transform
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, idx):
       # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        
        imgs, dst = self.datasets[dataset_idx][index]

        imgs = self.transform(imgs).to(torch.float16)
        
        return {
            'input': imgs,
            'target': dst.to(torch.float16)
        }

"""
L1 Reconstruction dataset

(DSCOVR L1 + ACE L2) corrupted -> DSCOVR L2 + ACE L2 not corrupted

forcing criterion based on base noise

"""
class ReconstructionDatasetL1(Dataset):
    def __init__(
            self, 
            scrap_date_list: list,
            sequence_length: timedelta = timedelta(days = 2),
            step_size: timedelta = timedelta(minutes = 5),
            drop_probability: float = 0.1,
            noise_factor: float = 0.1
    ):
        """
        Reconstruction Dataset for L1 

        # Output:
        ## Normal:
        DSCOVR L1 & L2 data
        ACE data
        ## Corrupted:
        DSCOVR L1 & L2 data
        ACE data
        """
        self.name_list = ['input', 'target']
        self.seq_len = sequence_length
        # Defining spacecrafts
        dscovr = DSCOVR()
        ace = ACE()
        ## scrap_date list
        scrap_dates = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp for l1 scalers
        l1_temp = []
        for scrap_date in scrap_dates:
                #DSCOVR & ACE 
                input_seq = [
                    *[tool.resample(step_size).mean() for tool in dscovr.MAGFC(scrap_date, 'both', True)],
                    ace.MAG(scrap_date),
                    ace.SIS(scrap_date),
                    ace.EPAM(scrap_date),
                    ace.SWEPAM(scrap_date)
                ] #creating the list of satellite data
                
                input_seq = np.concatenate(input_seq, axis = -1)

                self.datasets.append(ReconstructionDefault(input_seq, sequence_length, step_size))

                l1_temp.append(input_seq)
        #Transforms
        # Scaler 
        self.l1_scaler = StandardScaler().fit(np.concatenate(l1_temp, axis = 0))
        # Drop and noise
        self.transform = transforms.Compose(
            transforms.ToTensor(),
            RandomDropPixels(drop_probability),
            RandomNoise(noise_factor),
        )
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
        
        l1_sequence_normal = self.datasets[dataset_idx][index]
        
        l1_sequence_normal= self.l1_scaler(l1_sequence_normal)

        l1_sequence_corrupted = self.transform(l1_sequence_normal)

        return {
            'input': l1_sequence_normal,
            'target': l1_sequence_corrupted
        }