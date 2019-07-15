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

        """Check amount of flow data for each site and build a table of this"""
        self.siteyears = pd.DataFrame(index = self.signatures_frame.iloc[:, 0],
                                      columns = ['MinYear', 'MaxYear', 'NumYears', 'NumSamples', 'RunningTotal'])
        for idx_site in range(self.num_sites):
            flow_file = str(self.signatures_frame.iloc[idx_site, 0]) + '_streamflow_qc.txt'
            flow_data_name = os.path.join(self.root_dir_flow, flow_file)
            flow_data = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                    names=["year", "month", "day", "flow(cfs)", "qc"])
            """Find first/last year of data"""
            minyeartmp = flow_data.min(axis=0)["year"]
            maxyeartmp = min(flow_data.max(axis=0)["year"], maxgoodyear)
            numyearstmp = (maxyeartmp-minyeartmp+1)
            numsamplestmp = math.floor(numyearstmp/self.years_per_sample)
            if idx_site == 0:
                runningtotaltmp = numsamplestmp
            else:
                runningtotaltmp = runningtotaltmp + numsamplestmp

            """Write to table"""
            self.siteyears.loc[self.signatures_frame.iloc[idx_site, 0]] = \
                [minyeartmp, maxyeartmp, numyearstmp, numsamplestmp, runningtotaltmp]

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
        print("idx = ", idx, "idx_site = ", idx_site, ", idx_within_site = ", idx_within_site)

        """Get file names for climate and flow"""
        climate_file = str(self.signatures_frame.iloc[idx_site, 0]) + '_lump_cida_forcing_leap.txt'
        flow_file = str(self.signatures_frame.iloc[idx_site, 0]) +  '_streamflow_qc.txt'
        climate_data_name = os.path.join(self.root_dir_climate,climate_file)
        flow_data_name = os.path.join(self.root_dir_flow, flow_file)
        print("Got file names")

        """Extract correct years out of each file"""
        flow_data = pd.read_csv(flow_data_name, sep='\s+', skiprows=idx_within_site * self.years_per_sample * 365,
                                nrows=self.years_per_sample * 365, header=None, usecols=[1, 2, 3, 4, 5],
                                parse_dates=[[1, 2, 3]])
        flow_data.columns = ["date", "flow(cfs)", "qc"]
        print("Extracted Flow Data")

        #  Find years for flow data
        flow_date_start = flow_data.iloc[0, 0]
        flow_date_end = flow_date_start + pd.Timedelta('729 days')

        climate_data = pd.read_csv(climate_data_name, sep='\t', skiprows=4, header=None,
                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                   parse_dates=True,
                                   names=["date", "dayl(s)", "prcp(mm / day)", "srad(W / m2)", "swe(mm)",
                                          "tmax(C)", "tmin(C)", "vp(Pa)"])
        climate_data['date'] = pd.to_datetime(climate_data['date'], format='%Y %m %d %H') - pd.Timedelta('12 hours')
        print("Extracted Climate Data")

        climate_data = climate_data.loc[(climate_data['date'] >= flow_date_start) & \
                                        (climate_data['date'] <= flow_date_end)]
        climate_data = climate_data.reset_index(drop=True)

        """Missing data label converted to 0/1"""
        d = {'A': 1, 'A:e': 1, 'M': 0}
        flow_data["qc"] = flow_data["qc"].map(d)

        """Merge climate and flow into one array"""
        hyd_data = pd.concat([climate_data.drop('date', axis=1), flow_data.drop('date', axis=1)], axis=1)

        """Get signatures related to site"""
        signatures = self.signatures_frame.iloc[idx_site, 1:]
        signatures = np.array([signatures])
        signatures = signatures.astype('double').reshape(-1, 1)
        sample = {'hyd_data': hyd_data, 'signatures': signatures}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert pandas data frames in sample to Tensors."""

    def __call__(self, sample):
        hyd_data, signatures = sample['hyd_data'], sample['signatures']  # Extract components from input sample

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
        return [hyd_data_tensor,signatures_tensor]