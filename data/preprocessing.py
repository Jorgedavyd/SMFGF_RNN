from torchvision.datasets.utils import download_url
from datetime import datetime, timedelta,date
from sunpy.net import Fido, attrs as a
from viresclient import SwarmRequest
import matplotlib.pyplot as plt
from astropy import units as u
from bs4 import BeautifulSoup
import spacepy.pycdf as pycdf
from astropy.io import fits
import astropy.units as u
import xarray as xr
import pandas as pd
import numpy as np
import requests
import tarfile
import zipfile
import shutil
import shutil
import glob
import gzip
import csv
import os

"""
SWARM SPACECRAFT
"""

class SWARM:
    def MAG_x(self, scrap_date, sc = 'A'): #spacecrafts = ['A', 'B', 'C'] #scrap_date format YYYY-MM-DD
        try:
            csv_file_root = f'./data/SWARM/MAG{sc}/{scrap_date[0]}_{scrap_date[-1]}.csv'
            mag_x = pd.read_csv(csv_file_root, parse_dates = ['Timestamp'], index_col = 'Timestamp')
            mag_x.index = pd.to_datetime(mag_x.index)
            return mag_x
        except FileNotFoundError:
            request = SwarmRequest()
            # - See https://viresclient.readthedocs.io/en/latest/available_parameters.html
            request.set_collection(f"SW_OPER_MAG{sc}_LR_1B")
            request.set_products(
                measurements=[
                    'F',
                    'dF_Sun',
                    'B_VFM',
                    'dB_Sun',
                    ],
                auxiliaries=["Dst"],
            )
            # Fetch data from a given time interval
            # - Specify times as ISO-8601 strings or Python datetime
            data = request.get_between(start_time= scrap_date[0].isoformat(),end_time= scrap_date[-1].isoformat())
            df = data.as_dataframe()
            b_VFM = pd.DataFrame(df['B_VFM'].tolist(), columns=['B_VFM_1', 'B_VFM_2', 'B_VFM_3'], index = df.index)
            dB_Sun = pd.DataFrame(df['dB_Sun'].tolist(), columns=['dB_Sun_1', 'dB_Sun_2','dB_Sun_3'], index = df.index)
            df = pd.concat([df.drop(['B_VFM','dB_Sun','Spacecraft',], axis = 1), b_VFM, dB_Sun], axis = 1).resample('5T').mean()
            df.columns = ['Longitude', 'Dst','dF_Sun','F', 'Radius', 'Latitude', 'b_VFM_1', 'b_VFM_2', 'b_VFM_3', 'dB_Sun_1', 'dB_Sun_2','dB_Sun_3']
            os.makedirs(f'./data/SWARM/MAG{sc}', exist_ok = True)
            df.to_csv(csv_file_root)
            return df 
    def ION_x(self, scrap_date, sc = 'A'): #spacecrafts = ['A', 'B', 'C'] #scrap_date format YYYY-MM-DD
        try:
            csv_file_root = f'./data/SWARM/ion_plasma_{sc}/{scrap_date[0]}_{scrap_date[-1]}.csv'
            ion_x = pd.read_csv(csv_file_root, parse_dates = ['Timestamp'], index_col = 'Timestamp')
            return ion_x
        except FileNotFoundError:
            par_dict = {
                'Electric field instrument (Langmuir probe measurements at 2Hz)': [f"SW_OPER_EFI{sc}_LP_1B", ['Ne', 'Te', 'Vs','U_orbit']], #collection, measurements, 
                '16Hz cross-track ion flows': [f'SW_EXPT_EFI{sc}_TCT16', ['Ehx','Ehy','Ehz','Vicrx','Vicry','Vicrz']], 
                'Estimates of the ion temperatures': [f'SW_OPER_EFI{sc}TIE_2_',['Tn_msis', 'Ti_meas_drift']],
                '2Hz ion drift velocities and effective masses (SLIDEM project)': [f'SW_PREL_EFI{sc}IDM_2_', ['V_i','N_i', 'M_i_eff']]
            }
            df_list = []
            for parameters in par_dict.values():
                request = SwarmRequest()
                # - See https://viresclient.readthedocs.io/en/latest/available_parameters.html
                request.set_collection(parameters[0])
                request.set_products(
                    measurements=parameters[1]
                )
                # Fetch data from a given time interval
                # - Specify times as ISO-8601 strings or Python datetime
                data = request.get_between(
                    start_time= scrap_date[0] + 'T00:00',
                    end_time= scrap_date[-1] + 'T23:59',
                )
                df = data.as_dataframe().drop(['Longitude','Spacecraft','Radius','Latitude'], axis = 1)
                df.index = pd.to_datetime(df.index)
                df = df.resample('5T').mean() ##solve quality flags 
                df_list.append(df)
                del request
                del data
            os.makedirs(f'./data/SWARM/ion_plasma_{sc}', exist_ok = True)
            df = pd.concat(df_list, axis = 1)
            df.to_csv(csv_file_root)
            return df 

"""
WIND Spacecraft
"""

def WIND_MAG_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v4 = datetime.strptime('20230101', '%Y%m%d')
    v3 = datetime.strptime('20231121', '%Y%m%d')
    if date<v4:
        return 'v05'
    elif date<v3:
        return 'v04'
    else:
        return 'v03'
def WIND_SWE_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v4 = datetime.strptime('20230101', '%Y%m%d')
    v3 = datetime.strptime('20231121', '%Y%m%d')
    if date<v4:
        return 'v05'
    elif date<v3:
        return 'v04'
    else:
        return 'v03'
def TDP_PM_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v4 = datetime.strptime('20110114', '%Y%m%d')
    v5 = datetime.strptime('20111230', '%Y%m%d')
    if date<v4:
        return 'v03'
    elif date<v5:
        return 'v04'
    else:
        return 'v05'

class WIND:
    def MAG(self, scrap_date):
        try:
            csv_file = f'./data/WIND/MAG/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/MAG/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = ['BF1','BGSE','BGSM']
            variables = ['datetime', 'BF1'] + [f'{name}_{i}' for name in phy_obs[1:3] for i in range(1,4)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = WIND_MAG_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/mfi/mfi_h0/{date[:4]}/wi_h0_mfi_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:].squeeze(1)]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWE_alpha_proton(self, scrap_date): #includes spacecraft position
        try:
            csv_file = f'./data/WIND/SWE/alpha_proton/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/SWE/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['Proton_V_nonlin', 'Proton_VX_nonlin', 'Proton_VY_nonlin', 'Proton_VZ_nonlin', 'Proton_Np_nonlin', 'Alpha_V_nonlin', 'Alpha_VX_nonlin',
                         'Alpha_VY_nonlin', 'Alpha_VZ_nonlin', 'Alpha_Na_nonlin', 'xgse', 'ygse','zgse']
            variables = ['datetime'] + ['Vp', 'Vpx', 'Vpy', 'Vpz', 'Np', 'Va', 'Vax', 'Vay', 'Vaz', 'Na', 'xgse', 'ygse','zgse']
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/swe/swe_h1/{date[:4]}/wi_h1_swe_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWE_electron_angle(self, scrap_date):
        try:
            csv_file = f'./data/WIND/SWE/electron_angle/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/SWE/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['f_pitch_SPA','Ve'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h3_swe_00000000_v01.skt
            variables = ['datetime'] + [f'f_pitch_SPA_{i}' for i in range(13)] + [f'Ve_{i}' for i in range(13)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/swe/swe_h3/{date[:4]}/wi_h3_swe_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWE_electron_moments(self, scrap_date):
        try:
            csv_file = f'./data/WIND/SWE/electron_moments/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/SWE/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['N_elec','TcElec', 'U_eGSE', 'P_eGSE', 'W_elec', 'Te_pal'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime'] + phy_obs[:2] + [phy_obs[2] + f'_{i}' for i in range(1,4)]+ [phy_obs[3] + f'_{i}' for i in range(1,7)] + phy_obs[4:]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/swe/swe_h5/{date[:4]}/wi_h5_swe_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_PM(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/PM/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['P_VELS', 'P_TEMP','P_DENS','A_VELS','A_TEMP','A_DENS'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime', 'Vpx','Vpy','Vpz', 'Tp','Np','Vax', 'Vay', 'Vaz','Ta','Na'] #GSE
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = TDP_PM_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_pm/{date[:4]}/wi_pm_3dp_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_PLSP(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/PLSP/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9])
            phy_obs = ['FLUX', 'ENERGY', 'MOM.P.VTHERMAL', 'MOM.P.FLUX','MOM.P.PTENS', 'MOM.A.FLUX'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime'] + [f'FLUX_{i}'for i in range(1,16)]+ [f'ENERGY_{i}' for i in range(1,16)] + ['Vpt', 'Jpx', 'Jpy', 'Jpz','Pp_XX', 'Pp_YY', 'Pp_ZZ', 'Pp_XY', 'Pp_XZ', 'Pp_YZ', 'Jax', 'Jay', 'Jaz']
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_plsp/{date[:4]}/wi_plsp_3dp_{date}_v02.cdf'#https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_sfpd_3dp_00000000_v01.skt
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_SOSP(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/SOSP/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime'] + [phy_obs[i] + f'_{k}' for i in range(2) for k in range(1,10)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sosp/{date[:4]}/wi_sosp_3dp_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_SOPD(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/SOPD/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX'## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                "70keV",
                "130keV",
                "210keV",
                "330keV",
                "550keV",
                "1000keV",
                "2100keV",
                "4400keV",
                "6800keV"
            ]
            variables = ['datetime'] + [f'Proton_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands ]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sopd/{date[:4]}/wi_sopd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_ELSP(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/ELSP/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            energy_bands = [
                '1113eV' ,
                '669.2eV',
                '426.8eV',
                '264.8eV',
                '165eV'  ,
                '103.3eV',
                '65.25eV',
                '41.8eV' ,
                '27.25eV',
                '18.3eV',
                '12.8eV',
                '9.4eV' ,
                '7.25eV',
                '5.9eV' ,
                '5.2eV' 
            ]
            variables = ['datetime'] + [f'electron_{phy_obs[i]}_{ener}' for i in range(2) for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_elsp/{date[:4]}/wi_elsp_3dp_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_ELPD(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/ELPD/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX' ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elpd_3dp_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                '1150eV',
                '790eV',
                '540eV',
                '370eV',
                '255eV',
                '175eV',
                '121eV',
                '84eV',
                '58eV',
                '41eV',
                '29eV',
                '20.5eV',
                '15eV',
                '11.3eV',
                '8.6eV'
            ]
            variables = ['datetime'] + [f'electron_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_elpd/{date[:4]}/wi_elpd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_EHSP(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/EHSP/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_ehsp_3dp_00000000_v01.skt
            energy_bands = [
                '27660eV' ,
                '18940eV' ,
                '12970eV' ,
                '8875eV'  ,
                '6076eV'  ,
                '4161eV'  ,
                '2849eV'  ,
                '1952eV'  ,
                '1339eV'  ,
                '920.3eV',
                '634.4eV',
                '432.7eV',
                '292.0eV',
                '200.1eV',
                '136.8eV',
            ]
            variables = ['datetime'] + [f'electron_{phy_obs[i]}_{ener}' for i in range(2) for ener in energy_bands] 
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_ehsp/{date[:4]}/wi_ehsp_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_EHPD(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/EHPD/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX' ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_ehpd_3dp_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                '27660eV' ,
                '18940eV' ,
                '12970eV' ,
                '8875eV'  ,
                '6076eV'  ,
                '4161eV'  ,
                '2849eV'  ,
                '1952eV'  ,
                '1339eV'  ,
                '920.3eV',
                '634.4eV',
                '432.7eV',
                '292.0eV',
                '200.1eV',
                '136.8eV',
            ]
            variables = ['datetime'] + [f'electron_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_ehpd_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_ehpd/{date[:4]}/wi_ehpd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_SFSP(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/SFSP/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            energy_bands = [
                '27keV', 
                '40keV', 
                '86keV', 
                '110keV',
                '180keV',
                '310keV',
                '520keV'
            ]
            variables = ['datetime'] + [f'electron_{phy_obs[i]}_{ener}' for i in range(2) for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_sfsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sfsp/{date[:4]}/wi_sfsp_3dp_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_SFPD(self, scrap_date):
        try:
            csv_file = f'./data/WIND/TDP/SFPD/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX' ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elpd_3dp_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                '27keV', 
                '40keV', 
                '86keV', 
                '110keV',
                '180keV',
                '310keV',
                '520keV'
            ]
            variables = ['datetime'] + [f'electron_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sfpd/{date[:4]}/wi_sfpd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SMS(self, scrap_date):
        try:
            csv_file = f'./data/WIND/SMS/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/WIND/SMS/temp' 
            os.makedirs(temp_root)
            angle = [
                53,
                0,
                -53
            ]
            phy_obs = ['counts_tc_he2plus', 'counts_tc_heplus', 'counts_tc_hplus', 'counts_tc_o6plus', 'counts_tc_oplus', 'counts_tc_c5plus', 'counts_tc_fe10plus', 
                        'dJ_tc_he2plus', 'dJ_tc_heplus', 'dJ_tc_hplus', 'dJ_tc_o6plus', 'dJ_tc_oplus', 'dJ_tc_c5plus', 'dJ_tc_fe10plus']## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_l2-3min_sms-stics-vdf-solarwind_00000000_v01.skt
            variables = ['datetime'] + [phy_obs[i] + f'_{deg}'for i in range(len(phy_obs)) for deg in angle] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_l2-3min_sms-stics-vdf-solarwind_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/data/wind/sms/l2/stics_cdf/3min_vdf_solarwind/{date[:4]}/wi_l2-3min_sms-stics-vdf-solarwind_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    data_columns.append(np.mean(np.mean(cdf_file[var][:], axis = 1), axis=2))

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
"""
SOHO SPACECRAFT (ESA)
"""

class SOHO:
    def CELIAS_SEM(self, scrap_date):
        years = set(date[:4] for date in scrap_date)
        root = 'data/SOHO_L2/CELIAS_SEM_15sec_avg'
        csv_root = os.path.join(root, f'{scrap_date[0]}_{scrap_date[-1]}.csv')

        try:
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_SEM_15sec_avg.tar.gz'
            name = 'CELIAS_SEM_15sec_avg.tar.gz'
            os.makedirs(root, exist_ok=True)
            if f'{scrap_date[0]}_{scrap_date[-1]}.csv' in os.listdir(root):
                raise FileExistsError
            download_url(url, root, name)
            
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)
            
            with open(csv_root, 'w') as csv_file:
                csv_file.write("Julian,F_Flux,C_Flux\n")

            for year in years:
                data_rows = []  
                year_folder = os.path.join(root, year)
                for day in sorted(os.listdir(year_folder)):
                    file_path = os.path.join(year_folder, day)
                    with open(file_path, 'r') as txt:
                        lines = txt.readlines()[46:]  # Skip first 46 lines
                        for line in lines:
                            data = [line.split()[0]] + line.split()[-2:] #julian,flux
                            data_rows.append(data)
                
                with open(csv_root, 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for row in data_rows:
                        csv_writer.writerow(row)

                shutil.rmtree(year_folder)
            
            celias_sem = pd.read_csv(csv_root)
            celias_sem['datetime'] = pd.to_datetime(celias_sem['Julian'], unit='D', origin = 'julian')
            celias_sem.set_index('datetime', drop=True,inplace=True)
            celias_sem.drop('Julian', axis = 1, inplace = True)
            celias_sem.to_csv(csv_root)
        except FileExistsError:
            pass

        celias_sem = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%y-%m-%d %H:%M:%S')
        celias_sem.index = pd.to_datetime(celias_sem.index)
        celias_sem = celias_sem.sort_index()
        celias_sem = celias_sem.loc[scrap_date[0]: scrap_date[-1]]
        return celias_sem
    
    """CELIAS PROTON MONITOR"""
    """It has YY,MON,DY,DOY:HH:MM:SS,SPEED,Np,Vth,N/S,V_He"""
    def CELIAS_PM(self, scrap_date):
        csv_root = f'data/SOHO_L2/CELIAS_Proton_Monitor_5min/{scrap_date[0]}_{scrap_date[-1]}.csv'
        years = set(date[:4] for date in scrap_date)
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        
        try:
            root = 'data/SOHO_L2/CELIAS_Proton_Monitor_5min'
            name = 'CELIAS_Proton_Monitor_5min.tar.gz'
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min.tar.gz'
            os.makedirs(root, exist_ok=True)
            if f'{scrap_date[0]}_{scrap_date[-1]}.csv' in os.listdir(root):
                raise FileExistsError
            download_url(url, root, name)
            
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)

            with open(csv_root, 'w') as csv_file:
                csv_file.write("datetime,SPEED,Np,Vth,N/S,V_He\n")

            for year in years:
                data_rows = []
                filename = f'{year}_CELIAS_Proton_Monitor_5min'
                file_path = os.path.join(root, filename + '.zip')
                with zipfile.ZipFile(file_path, 'r') as archive:
                    with archive.open(filename+'.txt') as txt:
                        lines = txt.readlines() ####
                        for line in lines[29:]:
                            vector = [item.decode('utf-8') for item in line.split()]
                            data = [vector[0]+'-'+month_map[vector[1]]+'-'+vector[2]+' '+':'.join(vector[3].split(':')[1:])] + vector[4:-7] #ignores the position of SOHO
                            data_rows.append(data)
                with open(csv_root, 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for row in data_rows:
                        csv_writer.writerow(row)

                os.remove(file_path)
            
        except FileExistsError:
            pass

        df = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%y-%m-%d %H:%M:%S').sort_index()
        df = df.loc[scrap_date[0]:scrap_date[-1]]
        return df
    
    def COSTEP_EPHIN(self, scrap_date):
        #getting dates
        years = sorted(list(set([date[:4] for date in scrap_date]))) ##YYYMMDD
        root = './data/SOHO_L2/COSTEP_EPHIN_5min'
        csv_root = os.path.join(root, f'{scrap_date[0]}_{scrap_date[-1]}.csv')
        try:
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/COSTEP_EPHIN_L3_l3i_5min-EntireMission-ByYear.tar.gz'
            name = 'COSTEP_EPHIN_L3_l3i_5min-EntireMission-ByYear.tar.gz'
            os.makedirs(root, exist_ok=True)
            if f'{scrap_date[0]}_{scrap_date[-1]}.csv' in os.listdir(root):
                raise FileExistsError
            download_url(url, root, name)
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)
            
            columns = [
                'year', 'month', 'day', 'hour', 'minute',
                'int_p4', 'int_p8', 'int_p25', 'int_p41',
                'int_h4', 'int_h8', 'int_h25', 'int_h41'
            ]

            with open(csv_root, 'w') as csv_file:
                csv_file.writelines(','.join(columns) + '\n')

            for year in years:
                data_rows = []
                filename =os.path.join(root, '5min', f'{year}.l3i')
                with open(filename, 'r') as txt:
                    lines = txt.readlines()
                    for line in lines[3:]:
                        data = line.split()[:3] + line.split()[4:6] + line.split()[8:12] + line.split()[20:24]
                        data = [item for item in data]
                        data_rows.append(data)
                with open(csv_root, 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for row in data_rows:
                        csv_writer.writerow(row)
                os.remove(filename)
            shutil.rmtree(os.path.join(root, '5min'))

            df = pd.read_csv(csv_root)
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
            df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)  
            df.set_index('datetime', inplace=True, drop = True)
            df.to_csv(csv_root)
        except FileExistsError:
            pass
        costep_ephin = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%Y-%m-%d %H:%M:%S').sort_index()
        costep_ephin = costep_ephin.loc[scrap_date[0]:scrap_date[-1]]
        return costep_ephin
"""
ACE SPACECRAFT (ESA)
"""

def SIS_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v5 = datetime.strptime('20141104', '%Y%m%d')
    v6 = datetime.strptime('20171019', '%Y%m%d')
    if date<v5:
        return 'v04'
    elif date<v6:
        return 'v05'
    else:
        return 'v06'

def EPAM_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v5 = datetime.strptime('20150101', '%Y%m%d')
    if date<v5:
        return 'v04'
    else:
        return 'v05'

def MAG_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v5 = datetime.strptime('20030328', '%Y%m%d')
    v6 = datetime.strptime('20120630', '%Y%m%d')
    v7 = datetime.strptime('20180130', '%Y%m%d')
    if date<v5:
        return 'v04'
    elif date<v6:
        return 'v05'
    elif date<v7:
        return 'v06'
    else:
        return 'v07'

def SWEPAM_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v7 = datetime.strptime('20031030', '%Y%m%d')
    v8 = datetime.strptime('20050227', '%Y%m%d')
    v9 = datetime.strptime('20050325', '%Y%m%d')
    v10 = datetime.strptime('20061207', '%Y%m%d')
    v11 = datetime.strptime('20130101', '%Y%m%d')
    
    if date<v7:
        return 'v06'
    elif date<v8:
        return 'v07'
    elif date<v9:
        return 'v08'
    elif date<v10:
        return 'v09'
    elif date<v11:
        return 'v10'
    else:
        return 'v11'


## https://cdaweb.gsfc.nasa.gov/cgi-bin/eval1.cgi
class ACE:
    def SIS(self, scrap_date):
        try:
            csv_file = f'./data/ACE/SIS/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/ACE/SIS/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = [
                'flux_He', 'flux_C', 'flux_N', 'flux_O', 'flux_Ne', 'flux_Mg',
                'flux_Si', 'flux_S', 'flux_Ar', 'flux_Ca', 'flux_Fe',
                'flux_Ni', 'cnt_He', 'cnt_C', 'cnt_N', 'cnt_O', 'cnt_Ne',
                'cnt_Mg', 'cnt_Si', 'cnt_S', 'cnt_Ar', 'cnt_Ca',
                'cnt_Fe', 'cnt_Ni'
            ]
            variables = ['datetime'] + [f'{name}_{i}' for name in phy_obs for i in range(1,9)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = SIS_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/sis/level_2_cdaweb/sis_h1/{date[:4]}/ac_h1_sis_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def MAG(self, scrap_date):
        try:
            csv_file = f'./data/ACE/MAG/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
            temp_root = './data/ACE/MAG/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = ['Magnitude', 'BGSM', 'SC_pos_GSM', 'dBrms', 'BGSEc','SC_pos_GSE']
            variables = ['datetime'] + ['Bnorm', 'BGSM_x', 'BGSM_y', 'BGSM_z', 'SC_GSM_x', 'SC_GSM_y', 'SC_GSM_z', 'dBrms', 'BGSE_x', 'BGSE_y', 'BGSE_z', 'SC_GSE_x', 'SC_GSE_y', 'SC_GSE_z']
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = MAG_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/mag/level_2_cdaweb/mfi_h0/{date[:4]}/ac_h0_mfi_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWEPAM(self, scrap_date):
        csv_file = f'./data/ACE/SWEPAM/{scrap_date[0]}_{scrap_date[-1]}.csv' #directories
        try:
            temp_root = './data/ACE/SWEPAM/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = ['Np', 'Vp','Tpr','alpha_ratio', 'V_GSE', 'V_GSM'] #variables#change
            variables = ['datetime'] + phy_obs[:4] + [phy_obs[i] + f'_{k}' for i in range(4,6) for k in range(1,4)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = SWEPAM_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/swepam/level_2_cdaweb/swe_h0/{date[:4]}/ac_h0_swe_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)
                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWICS(self, scrap_date):
        name_lambda = lambda date: f'./data/.../{date}.csv'  
        temp_root = './data/ACE/SWICS/temp' 
        os.makedirs(temp_root) #create folder
        phy_obs = ['nH', 'vH','vthH'] #variables#change
        variables = ['datetime'] + phy_obs
        for date in scrap_date:
            csv_file = name_lambda(date) #directories
            try:
                with open(csv_file, 'x') as file:
                    file.writelines(','.join(variables) + '\n')
            except FileExistsError:
                continue
            url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/swics/level_2_cdaweb/swi_h6/{date[:4]}/ac_h6_swi_{date}_v03.cdf'
            name = date + '.cdf'
            download_url(url, temp_root, name)
            cdf_path = os.path.join(temp_root, name)
            cdf_file = pycdf.CDF(cdf_path)
            data_columns = []

            for var in phy_obs:
                cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                data_columns.append(cond)

            epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
            data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
            data  = np.concatenate([epoch, data], axis = 1)
            with open(csv_file, 'a') as file:
                np.savetxt(file, data, delimiter=',', fmt='%s')
            cdf_file.close()
        shutil.rmtree(temp_root)
        data = [pd.read_csv(name_lambda(date), parse_dates=['datetime'], index_col = 'datetime') for date in scrap_date]
        init_date = pd.to_datetime(scrap_date[0], format = '%Y%m%d')
        end_date  = pd.to_datetime(scrap_date[-1], format = '%Y%m%d')
        df = pd.concat(data, axis = 0).loc[init_date: end_date]
        return df
    def EPAM(self, scrap_date):
        name_lambda = lambda date: f'./data/ACE/EPAM/{date}.csv'
        temp_root = './data/ACE/EPAM/temp' 
        phy_obs = [
                    'P7',
                    'P8',
                    'DE1',
                    'DE2',
                    'DE3',
                    'DE4',
                    'W3',
                    'W4',
                    'W5',
                    'W6',
                    'W7',
                    'W8',
                    'E1p',
                    'E2p',
                    'E3p',
                    'E4p',
                    'FP5p',
                    'FP6p',
                    'FP7p',
                    'Z2',
                    'Z2A',
                    'Z3',
                    'Z4',
                    'P1p',
                    'P2p',
                    'P3p',
                    'P4p',
                    'P5p',
                    'P6p',
                    'P7p',
                    'P8p',
                    'E4',
                    'FP5',
                    'FP6',
                    'FP7'] #All available readings
        os.makedirs(temp_root) #create folder
        variables = ['datetime'] + phy_obs#variables#change
        for date in scrap_date:
            csv_file = name_lambda(date) #directories
            try:
                with open(csv_file, 'x') as file:
                    file.writelines(','.join(variables) + '\n')
            except FileExistsError:
                continue
            version = EPAM_version(date)
            url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/epam/level_2_cdaweb/epm_h1/{date[:4]}/ac_h1_epm_{date}_{version}.cdf'
            name = date + '.cdf'
            download_url(url, temp_root, name)
            cdf_path = os.path.join(temp_root, name)
            cdf_file = pycdf.CDF(cdf_path)
            data_columns = []

            for var in phy_obs:
                cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                data_columns.append(cond)

            epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
            data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
            data  = np.concatenate([epoch, data], axis = 1)
            with open(csv_file, 'a') as file:
                np.savetxt(file, data, delimiter=',', fmt='%s')
            cdf_file.close()
        shutil.rmtree(temp_root)
        data = [pd.read_csv(name_lambda(date), parse_dates=['datetime'], index_col = 'datetime') for date in scrap_date]
        init_date = pd.to_datetime(scrap_date[0], format = '%Y%m%d')
        end_date  = pd.to_datetime(scrap_date[-1], format = '%Y%m%d')
        df = pd.concat(data, axis = 0).loc[init_date: end_date]
        return df

"""
DSCOVR Spacecraft
"""
class DSCOVR:
    def MAGFC(self, scrap_date: list, level = None, joint = False):
        scrap_date = [string.replace('-','') for string in scrap_date]
        os.makedirs('data/compressed', exist_ok=True)
        os.makedirs('data/uncompressed', exist_ok=True)
        os.makedirs('data/DSCOVR_L2/faraday', exist_ok=True)
        os.makedirs('data/DSCOVR_L1/faraday', exist_ok=True)
        os.makedirs('data/DSCOVR_L2/magnetometer', exist_ok=True)
        os.makedirs('data/DSCOVR_L1/magnetometer', exist_ok=True)
        with open('data/URLs.csv', 'r') as file:
            lines = file.readlines()
        url_list = []
        for url in lines:
            for date in scrap_date:
                if date+'000000' in url:
                    url_list.append(url)
        for url in url_list:
            root = 'data/compressed'
            filename = url.split('_')[1] + '_'+url.split('_')[3][1:-6]+'.nc.gz'
            if (filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L2/faraday')) or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L1/faraday') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L1/magnetometer') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L2/magnetometer'):
                continue
            elif filename[:-3] in os.listdir('data/uncompressed'):
                output_file = os.path.join('data/uncompressed',filename)[:-3]
                pass
            else:
                download_url(url, root, filename)
                file = os.path.join(root,filename)
                output_file = os.path.join('data/uncompressed',filename)[:-3]
                with gzip.open(file, 'rb') as compressed_file:
                        with open(output_file, 'wb') as decompressed_file:
                            decompressed_file.write(compressed_file.read())
                os.remove(file)
            if 'fc1' in filename:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['proton_density', 'proton_speed', 'proton_temperature']

                faraday_cup = df[important_variables]

                faraday_cup = faraday_cup.resample('1min').mean()

                faraday_cup.to_csv(f'data/DSCOVR_L1/faraday/{filename[:-6]}.csv')

                os.remove(output_file)

            elif 'f1m' in filename:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['proton_density', 'proton_speed', 'proton_temperature']

                faraday_cup = df[important_variables]
                ##Feature engineering

                faraday_cup.to_csv(f'data/DSCOVR_L2/faraday/{filename[:-6]}.csv')
                
                os.remove(output_file)
            elif 'm1m' in filename:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']

                magnetometer = df[important_variables]

                magnetometer.to_csv(f'data/DSCOVR_L2/magnetometer/{filename[:-6]}.csv')
                
                os.remove(output_file)
            else:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']

                magnetometer = df[important_variables]

                magnetometer = magnetometer.resample('1min').mean()

                magnetometer.to_csv(f'data/DSCOVR_L1/magnetometer/{filename[:-6]}.csv')
                os.remove(output_file)

        return self.from_csv(scrap_date, level, joint)

    def from_csv(self, scrap_date, level = 'both', joint = False):
        
        fc1_list = []
        mg1_list = []
        f1m_list = []
        m1m_list = []
        for ind_date in scrap_date:

            fc1_list.append(pd.read_csv(f'./data/DSCOVR_L1/faraday/fc1_{ind_date}.csv', index_col=0)[['proton_density', 'proton_speed', 'proton_temperature']])
            mg1_list.append(pd.read_csv(f'./data/DSCOVR_L1/magnetometer/mg1_{ind_date}.csv', index_col=0)[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']])
            f1m_list.append(pd.read_csv(f'./data/DSCOVR_L2/faraday/f1m_{ind_date}.csv', index_col=0)[['proton_density', 'proton_speed', 'proton_temperature']])
            m1m_list.append(pd.read_csv(f'./data/DSCOVR_L2/magnetometer/m1m_{ind_date}.csv', index_col=0)[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']])

        f1m = pd.concat(f1m_list)
        m1m = pd.concat(m1m_list)
        f1m.index = pd.to_datetime(f1m.index)
        m1m.index = pd.to_datetime(m1m.index)
        freq = '5T'
        f1m = f1m.resample(freq).mean()
        m1m = m1m.resample(freq).mean()
        if level == 'both':
            fc1 = pd.concat(fc1_list)
            mg1 = pd.concat(mg1_list)
            fc1.index = pd.to_datetime(fc1.index)
            mg1.index = pd.to_datetime(mg1.index)
            fc1= f1m.resample(freq).mean()
            mg1= m1m.resample(freq).mean()
            return pd.concat([fc1, mg1], axis = 1), pd.concat([f1m, m1m], axis = 1) if joint else fc1,mg1,f1m,m1m
        else:
            return pd.concat([f1m, m1m]) if joint else f1m,m1m

def interval_time(start_date_str, end_date_str, format = "%Y%m%d"):
    start_date = datetime.strptime(start_date_str, format)
    end_date = datetime.strptime(end_date_str, format)

    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime(format))
        current_date += timedelta(days=1)

    return date_list

def Dst(scrap_date, resample_method = '5T'):
    try:        
        data_list = []
        for day in scrap_date:
            today_dst = pd.read_csv(f'data/Dst_index/{day[:6]}.csv',index_col = 0, header = None).T[int(day[6:])]
            for i,k in enumerate(today_dst):
                if isinstance(k, str): 
                    today_dst[i+1] = float(today_dst[i+1])
                if np.abs(today_dst[i+1])>500:
                    today_dst[i+1] = np.nan
            
            data_list.append(today_dst)
        series = pd.concat(data_list, axis = 0)

        series.index = pd.date_range(scrap_date[0]+ ' 00:00:00', scrap_date[-1] + ' 23:59:59', freq = '1H')
        series = series.resample(resample_method).ffill()
        series.name = 'Dst'
        return series
    except FileNotFoundError:
        os.makedirs('data/Dst_index', exist_ok = True)
        month = day[:6]

        if int(str(month)[:4])<=int(date.today().year) - 1:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{month}/index.html'
        elif 2017<=int(str(month)[:4])<int(date.today().year)-1:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{month}/index.html'
        else:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_final/{month}/index.html'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.text

            soup = BeautifulSoup(data, 'html.parser')
            data = soup.find('pre', class_='data')
            with open('data/Dst_index/'+ url.split('/')[-2]+'.csv', 'w') as file:
                file.write('\n'.join(data.text.replace('\n\n', '\n').replace('\n ','\n').split('\n')[7:39]).replace('-', ' -').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            Dst(scrap_date)
        else:
            print('Unable to access the site')

class SDO:
    def AIA_HMI(self, scrap_date, sample_int: int = 5):
        """
        Flares:
        
        SDO/AIA 94
        SDO/AIA 131

        CMEs:

        SDO/AIA 6173 Å
        SDO/AIA 1600 Å
        SDO/AIA 171 Å
        SDO/AIA 193 Å

        """
        try:
            query = glob.glob(f'./data/SDO/Joint/*{date_to_name(scrap_date)}**')
            if len(query) >= 1:
                return query
            else:
                raise FileNotFoundError
               
        except FileNotFoundError:
            
            wavelengths = [94, 131, 1600, 171, 193]
            files = {}
            for wavelength in wavelengths:
                result = Fido.search(
                    a.Time(scrap_date[0], scrap_date[-1]),
                    a.Instrument('AIA'),
                    a.Wavelength(wavelength * u.Angstrom),
                    a.Sample(sample_int*u.min),
                )
                files[wavelength] = Fido.fetch(result, path =  './data/SDO/AIA', progress = False)
            
            files[6000] = Fido.fetch(
                Fido.search(
                    a.Time(scrap_date[0], scrap_date[-1]),
                    a.Instrument('HMI'),
                    a.Wavelength(6000 * u.Angstrom),
                    a.Sample(sample_int*u.min),
                ),
                path =  './data/SDO/HMI', progress = False
            )
            file_paths = lambda x: files.values()[:][x]
            for i in range(len(file_paths(0))):
                data = []
                for file_name in file_paths(i):
                    with fits.open(file_name) as hdulist:
                        # Access the data array from the primary HDU
                        data.append(hdulist[0].data)
                plt.imsave(f'./data/SDO/Joint/{date_to_name(scrap_date)(i)}', np.concatenate(data, axis = 2))
    
def date_to_name(scrap_date: datetime):
    init_date = datetime.strftime(scrap_date[0], 'YYYY-MM-DD')
    last_date = datetime.strftime(scrap_date[-1], 'YYYY-MM-DD')

    return lambda x: init_date +'_'+ last_date + str(x) + '.png'

def scrap_date_to_month(scrap_date):
    return tuple(set([(day[4:6], day[:4]) for day in scrap_date]))
class OMNI_Data:
    def HRO2(self, scrap_date):
        months, years = scrap_date_to_month(scrap_date)
        temp_root = './data/OMNI/HRO2/temp' 
        os.makedirs(temp_root) #create folder
        phy_obs = ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'Mach_num', 'Mgs_mach_num', 'PR-FLX_10', 'PR-FLX_30', 'PR-FLX_60', 
        'proton_density', 'flow_speed', 'Vx', 'Vy', 'Vz'] ## metadata:https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/omni_hro2_5min_00000000_v01.skt
        variables = ['datetime'] + phy_obs #GSE
        #Instantiate the dates
        for year in years:
            for month in months:
                date = f'{year}-{month}'
                csv_file = f'./data/OMNI/HRO2/{date}.csv' #directories
                try:
                    with open(csv_file, 'x') as file:
                        file.writelines(','.join(variables) + '\n')
                except FileExistsError:
                    continue
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/omni/hro2_5min/{date[:4]}/omni_hro2_5min_{date[:6]}01_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
                cdf_file.close()
        shutil.rmtree(temp_root)
        data = []
        for year in years:
            for month in months:
                file_csv = f'./data/OMNI/HRO2/{year}-{month}.csv'
                data.append(pd.read_csv(file_csv, parse_dates = ['datetime'], index_col = 'datetime'))
        
        df = pd.concat(data, axis = 0).loc[pd.Timestamp(scrap_date[0]): pd.Timestamp(scrap_date[-1])]
        return df


def CDAWebDefault(scrap_date, name_lambda, phy_obs, variables, url_lambda):
        temp_root = name_lambda('')[:-4] + 'temp'
        os.makedirs(temp_root, exist_ok=True)
        variables = ['datetime'] + variables
        for date in scrap_date:
            csv_file = name_lambda(date) #directories
            try:
                with open(csv_file, 'x') as file:
                    file.writelines(','.join(variables) + '\n')
            except FileExistsError:
                continue
            url = url_lambda(date)
            name = date + '.cdf'
            download_url(url, temp_root, name)
            cdf_path = os.path.join(temp_root, name)
            cdf_file = pycdf.CDF(cdf_path)
            data_columns = []

            for var in phy_obs:
                cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                data_columns.append(cond)

            epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
            data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
            data  = np.concatenate([epoch, data], axis = 1)
            with open(csv_file, 'a') as file:
                np.savetxt(file, data, delimiter=',', fmt='%s')
            cdf_file.close()
        shutil.rmtree(temp_root)
        data = [pd.read_csv(name_lambda(date), parse_dates=['datetime'], index_col = 'datetime') for date in scrap_date]
        init_date = pd.to_datetime(scrap_date[0], format = '%Y%m%d')
        end_date  = pd.to_datetime(scrap_date[-1], format = '%Y%m%d')
        df = pd.concat(data, axis = 0).loc[init_date: end_date]
        return df
