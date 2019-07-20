from __future__ import print_function, division
import os
import torch
import math
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import datetime

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""Used https://pytorch.org/tutorials/beginner/data_loading_tutorial.html as example"""

class CamelsDataset(Dataset):
    """CAMELS dataset."""

    def __init__(self, csv_file, root_dir_climate, root_dir_flow, years_per_sample,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with signatures.
               needs to be a csv file with catchment name followed by the (numeric?) signatures
               data files should then be named as the catchment name (includes extension)
            root_dir (string): Directory with all the rain, ET, flow data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        maxgoodyear = 2013  # Don't use any years after this

        col_names = pd.read_csv(csv_file, nrows=0).columns
        types_dict = {'gauge_id': str}
        types_dict.update({col: float for col in col_names if col not in types_dict})

        self.signatures_frame = pd.read_csv(csv_file, sep=';', dtype=types_dict)
        self.root_dir_climate = root_dir_climate
        self.root_dir_flow = root_dir_flow
        self.transform = transform
        self.years_per_sample = years_per_sample

        """number of samples depends on years/sample. 35 years total; use up to 34 as 35 has a lot of missing data"""
        self.num_sites = len(self.signatures_frame)
        self.num_samples_per_site = math.floor(34 / self.years_per_sample)
        self.num_samples = self.num_sites * self.num_samples_per_site

        """Save areas in order to later convert flow data to mm"""
        root_dir_signatures = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'camels_attributes_v2.0')
        area_file = os.path.join(root_dir_signatures, 'camels_topo.txt')
        col_names = pd.read_csv(area_file, nrows=0).columns
        types_dict = {'gauge_id': str}
        types_dict.update({col: float for col in col_names if col not in types_dict})
        area_data = pd.read_csv(area_file, sep=';', dtype=types_dict)
        area_data = pd.DataFrame(area_data[['gauge_id', 'area_gages2']])
        area_data.set_index('gauge_id',inplace=True)
        self.area_data = area_data

        """Check amount of flow data for each site and build a table of this"""
        self.siteyears = pd.DataFrame(index = self.signatures_frame.iloc[:, 0],
                                      columns = ['MinYear', 'MaxYear', 'NumYears', 'NumSamples', 'RunningTotal',
                                                 'Flowmean_mmd','flow_std',
                                                 "dayl_av", "prcp_av", "srad_av", "swe_av",
                                                 "tmax_av", "tmin_av", "vp_av",
                                                 "dayl_std", "prcp_std", "srad_std", "swe_std",
                                                 "tmax_std", "tmin_std", "vp_std"])
        for idx_site in range(self.num_sites):
            """Read in climate and flow data for this site"""
            gauge_id = str(self.signatures_frame.iloc[idx_site, 0])
            flow_file = gauge_id + '_streamflow_qc.txt'
            flow_data_name = os.path.join(self.root_dir_flow, flow_file)
            flow_data = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                    names=["year", "month", "day", "flow(cfs)", "qc"])
            climate_file = str(self.signatures_frame.iloc[idx_site, 0]) + '_lump_cida_forcing_leap.txt'
            climate_data_name = os.path.join(self.root_dir_climate, climate_file)
            climate_data = pd.read_csv(climate_data_name, sep='\t', skiprows=4, header=None,
                                       usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                       parse_dates=True,
                                       names=["date", "dayl(s)", "prcp(mm / day)", "srad(W / m2)", "swe(mm)",
                                              "tmax(C)", "tmin(C)", "vp(Pa)"])

            """Find first/last year of data"""
            minyeartmp = flow_data[(flow_data["month"]==1)&(flow_data["day"]==1)].min(axis=0)["year"]
            maxyeartmp = flow_data[(flow_data["month"]==12)&(flow_data["day"]==30)].max(axis=0)["year"]
            maxyeartmp = min(maxyeartmp,maxgoodyear)
            numyearstmp = (maxyeartmp-minyeartmp+1)
            numsamplestmp = math.floor(numyearstmp/self.years_per_sample)
            if idx_site == 0:
                runningtotaltmp = numsamplestmp
            else:
                runningtotaltmp = runningtotaltmp + numsamplestmp

            """Calculate flow data average"""
            flow_data_mmd_mean = np.nanmean(flow_data['flow(cfs)'])*0.101947/self.area_data.loc[gauge_id, 'area_gages2']
            flow_std = np.nanstd(flow_data['flow(cfs)']) * 0.101947 / self.area_data.loc[
                gauge_id, 'area_gages2']

            """Calculate climate data mean and std"""
            dayl_mean = np.nanmean(climate_data["dayl(s)"])
            prcp_mean = np.nanmean(climate_data["prcp(mm / day)"])
            srad_mean = np.nanmean(climate_data["srad(W / m2)"])
            swe_mean = np.nanmean(climate_data["swe(mm)"])
            tmax_mean = np.nanmean(climate_data["tmax(C)"])
            tmin_mean = np.nanmean(climate_data["tmin(C)"])
            vp_mean = np.nanmean(climate_data["vp(Pa)"])
            dayl_std = np.nanstd(climate_data["dayl(s)"])
            prcp_std = np.nanstd(climate_data["prcp(mm / day)"])
            srad_std = np.nanstd(climate_data["srad(W / m2)"])
            swe_std = np.nanstd(climate_data["swe(mm)"])
            tmax_std = np.nanstd(climate_data["tmax(C)"])
            tmin_std = np.nanstd(climate_data["tmin(C)"])
            vp_std = np.nanstd(climate_data["vp(Pa)"])

            """Write to table"""
            self.siteyears.loc[self.signatures_frame.iloc[idx_site, 0]] = \
                [minyeartmp, maxyeartmp, numyearstmp, numsamplestmp, runningtotaltmp, flow_data_mmd_mean, flow_std,
                 dayl_mean, prcp_mean, srad_mean, swe_mean, tmax_mean, tmin_mean, vp_mean,
                 dayl_std, prcp_std, srad_std, swe_std, tmax_std, tmin_std, vp_std]

        """Normalization of signatures"""
        sig_norm = self.signatures_frame.iloc[:, 1:]
        sig_norm_mean = np.nanmean(sig_norm,axis=0)
        sig_norm_std = np.nanstd(sig_norm,axis=0)
        self.signatures_frame.iloc[:, 1:] = ((self.signatures_frame.iloc[:, 1:] - sig_norm_mean)/sig_norm_std)

        """Normalization of flow data"""
        overall_mean = np.nanmean(self.siteyears['Flowmean_mmd'])
        overall_std = (np.nanmean((self.siteyears['flow_std'])**2 +
                                  (self.siteyears['Flowmean_mmd']-overall_mean)**2))**(1/2.0)

        self.flow_norm = pd.DataFrame(np.array([[overall_mean, overall_std]]),
                                    index=['flow'],
                                    columns=['mean', 'std'])

        """Normalization of climate data"""
        dayl_overall_mean = np.nanmean(self.siteyears['dayl_av'])
        dayl_overall_std = (np.nanmean((self.siteyears['dayl_std']) ** 2 +
                                  (self.siteyears['dayl_av'] - dayl_overall_mean) ** 2)) ** (1 / 2.0)
        prcp_overall_mean = np.nanmean(self.siteyears['prcp_av'])
        prcp_overall_std = (np.nanmean((self.siteyears['prcp_std']) ** 2 +
                                       (self.siteyears['prcp_av'] - prcp_overall_mean) ** 2)) ** (1 / 2.0)
        srad_overall_mean = np.nanmean(self.siteyears['srad_av'])
        srad_overall_std = (np.nanmean((self.siteyears['srad_std']) ** 2 +
                                   (self.siteyears['srad_av'] - srad_overall_mean) ** 2)) ** (1 / 2.0)
        swe_overall_mean = np.nanmean(self.siteyears['swe_av'])
        swe_overall_std = (np.nanmean((self.siteyears['swe_std']) ** 2 +
                                   (self.siteyears['swe_av'] - swe_overall_mean) ** 2)) ** (1 / 2.0)
        tmax_overall_mean = np.nanmean(self.siteyears['tmax_av'])
        tmax_overall_std = (np.nanmean((self.siteyears['tmax_std']) ** 2 +
                                   (self.siteyears['tmax_av'] - tmax_overall_mean) ** 2)) ** (1 / 2.0)
        tmin_overall_mean = np.nanmean(self.siteyears['tmin_av'])
        tmin_overall_std = (np.nanmean((self.siteyears['tmin_std']) ** 2 +
                                   (self.siteyears['tmin_av'] - tmin_overall_mean) ** 2)) ** (1 / 2.0)
        vp_overall_mean = np.nanmean(self.siteyears['vp_av'])
        vp_overall_std = (np.nanmean((self.siteyears['vp_std']) ** 2 +
                                   (self.siteyears['vp_av'] - vp_overall_mean) ** 2)) ** (1 / 2.0)


        self.climate_norm = pd.DataFrame(np.array([[dayl_overall_mean,dayl_overall_std],
                                   [prcp_overall_mean,prcp_overall_std],
                                   [srad_overall_mean,srad_overall_std],
                                   [swe_overall_mean, swe_overall_std],
                                   [tmax_overall_mean, tmax_overall_std],
                                   [tmin_overall_mean, tmin_overall_std],
                                   [vp_overall_mean, vp_overall_std]]),
                                    index=['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'],
                                    columns=['mean', 'std'])

        self.num_samples = runningtotaltmp



    def __len__(self):

        return self.num_samples

    def __getitem__(self, idx):
        """Allow for each site corresponding to multiple samples"""
        idx_site = self.siteyears.index.get_loc(self.siteyears.index[self.siteyears['RunningTotal'] > idx][0])
        if idx_site == 0:
            idx_within_site = idx
        else:
            idx_within_site = idx - self.siteyears.iloc[idx_site - 1, 4]
        #  print("idx = ", idx, "idx_site = ", idx_site, ", idx_within_site = ", idx_within_site)

        """Get file names for climate and flow"""
        gauge_id = str(self.signatures_frame.iloc[idx_site, 0])
        climate_file = gauge_id + '_lump_cida_forcing_leap.txt'
        flow_file = gauge_id +  '_streamflow_qc.txt'
        climate_data_name = os.path.join(self.root_dir_climate,climate_file)
        flow_data_name = os.path.join(self.root_dir_flow, flow_file)
        #  print("Got file names")

        """Extract correct years out of each file"""
        flow_data_ymd = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                names=["year", "month", "day", "flow(cfs)", "qc"])
        flow_data = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                parse_dates=[[1, 2, 3]])
        flow_data.columns = ["date", "flow(cfs)", "qc"]
        # convert to float
        flow_data["flow(cfs)"] = flow_data["flow(cfs)"].astype(float)

        minyeartmp = flow_data_ymd[(flow_data_ymd["month"] == 1) & (flow_data_ymd["day"] == 1)].min(axis=0)["year"]
        minyearidx = minyeartmp + idx_within_site * self.years_per_sample
        #  Find years for flow data
        flow_date_start = pd.datetime(minyearidx, 1, 1)
        flow_date_end = flow_date_start + pd.Timedelta('729 days')

        flow_data = flow_data.loc[(flow_data['date'] >= flow_date_start) & \
                                        (flow_data['date'] <= flow_date_end)]
        flow_data = flow_data.reset_index(drop=True)

        #  print("Extracted Flow Data")

        climate_data = pd.read_csv(climate_data_name, sep='\t', skiprows=4, header=None,
                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                   parse_dates=True,
                                   names=["date", "dayl(s)", "prcp(mm / day)", "srad(W / m2)", "swe(mm)",
                                          "tmax(C)", "tmin(C)", "vp(Pa)"])
        climate_data['date'] = pd.to_datetime(climate_data['date'], format='%Y %m %d %H') - pd.Timedelta('12 hours')
        #  print("Extracted Climate Data")

        climate_data = climate_data.loc[(climate_data['date'] >= flow_date_start) & \
                                        (climate_data['date'] <= flow_date_end)]
        climate_data = climate_data.reset_index(drop=True)

        """Missing data label converted to 0/1"""
        d = {'A': 1, 'A:e': 1, 'M': 0}
        flow_data["qc"] = flow_data["qc"].map(d)
        flow_data["qc"][np.isnan(flow_data["qc"])] = 1

        """Normalize climate data"""
        climate_data["dayl(s)"] = ((climate_data["dayl(s)"] - self.climate_norm.loc["dayl","mean"])/
                                   self.climate_norm.loc["dayl","std"])
        climate_data["prcp(mm / day)"] = ((climate_data["prcp(mm / day)"])/
                                          self.climate_norm.loc["prcp","std"])
        climate_data["srad(W / m2)"] = ((climate_data["srad(W / m2)"] - self.climate_norm.loc["srad","mean"])/
                                        self.climate_norm.loc["srad","std"])
        climate_data["swe(mm)"] = ((climate_data["swe(mm)"] - self.climate_norm.loc["swe","mean"])/
                                   self.climate_norm.loc["swe","std"])
        climate_data["tmax(C)"] = ((climate_data["tmax(C)"] - self.climate_norm.loc["tmax","mean"])/
                                   self.climate_norm.loc["tmax","std"])
        climate_data["tmin(C)"] = ((climate_data["tmin(C)"] - self.climate_norm.loc["tmin","mean"])/
                                   self.climate_norm.loc["tmin","std"])
        climate_data["vp(Pa)"] = ((climate_data["vp(Pa)"] - self.climate_norm.loc["vp","mean"])/
                                  self.climate_norm.loc["vp","std"])

        """Normalize flow data"""
        """First to mm/d"""
        flow_area = self.area_data.loc[gauge_id, 'area_gages2']
        flow_data["flow(cfs)"] = flow_data["flow(cfs)"] * 0.101947 / flow_area
        """Then normalize"""
        flow_data["flow(cfs)"] = ((flow_data["flow(cfs)"] - self.flow_norm.iloc[0,0])/self.flow_norm.iloc[0,1])

        """Merge climate and flow into one array"""
        hyd_data = pd.concat([climate_data.drop(['date', 'swe(mm)'], axis=1), flow_data.drop('date', axis=1)], axis=1)
        if hyd_data.isnull().any().any():
            print('nan in hyd data')

        """Get signatures related to site"""
        signatures = self.signatures_frame.iloc[idx_site, 1:]
        signatures = np.array([signatures])
        signatures = signatures.astype('double').reshape(-1, 1)
        if np.isnan(signatures).any():
            print('nan in signatures')

        sample = {'gauge_id': gauge_id, 'date_start': str(flow_date_start), 'hyd_data': hyd_data, 'signatures': signatures}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert pandas data frames in sample to Tensors."""

    def __call__(self, sample):
        gauge_id, date_start, hyd_data, signatures = sample['gauge_id'], sample['date_start'], sample['hyd_data'], \
                                                     sample['signatures']
        # Extract components from input sample

        # swap axes because
        # pandas data: H x C
        # torch data: C X H
        hyd_data = hyd_data.transpose()
        hyd_data_values = hyd_data.values.astype(np.float64)
        """return {'hyd_data': torch.from_numpy(hyd_data_values),
                'signatures': torch.from_numpy(signatures)}"""
        hyd_data_tensor = torch.from_numpy(hyd_data_values)
        hyd_data_tensor.double()
        signatures_tensor = torch.from_numpy(signatures)
        signatures_tensor.double()
        return [gauge_id, date_start, hyd_data_tensor,signatures_tensor]