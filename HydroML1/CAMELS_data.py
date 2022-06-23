from __future__ import print_function, division
import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

from Util import *
from DataPoint import *

cfs2mm = 2.446575546

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


"""Used https://pytorch.org/tutorials/beginner/data_loading_tutorial.html as example"""


class CamelsDataset(Dataset):
    """CAMELS dataset."""

    def __init__(self, gauge_id_file, root_dir_climate, root_dir_camels_attributes, root_dir_flow,
                 dataset_properties: DatasetProperties, subsample_data, ablation_train=False, ablation_validate=False,
                 gauge_id=None, num_years=-1):

        self.gauge_id_file = pd.read_csv(gauge_id_file, dtype=str, sep='\t')
        self.gauge_id_file['gauge_id'] = self.gauge_id_file['gauge_id'].apply(lambda gauge_id: gauge_id if len(gauge_id) == 8 else ('0'+str(gauge_id)))
        #self.gauge_id_file['gauge_id'] = self.gauge_id_file['gauge_id'].strip()

        self.normalize_inputs = False
        self.normalize_outputs = False

        csv_file_attrib = [os.path.join(root_dir_camels_attributes, 'camels_' + s + '.txt') for s in
                           ['soil', 'topo', 'vege', 'geol']]

        csv_file = os.path.join(root_dir_camels_attributes, 'camels_hydro.txt')

        types_dict = {'gauge_id': str}
        self.attrib_files = CamelsDataset.read_attributes(csv_file_attrib, ';')
        self.latlong = self.attrib_files[['gauge_id', 'gauge_lat', 'gauge_lon']]
        self.attrib_files = self.attrib_files[['gauge_id'] + list(dataset_properties.attrib_normalizers.keys())]

        CamelsDataset.remove_nan(self.attrib_files)

        for name, normalizer in dataset_properties.attrib_normalizers.items():
            self.attrib_files[name] = self.attrib_files[name].transform(lambda x: x*normalizer)

        col_names = pd.read_csv(csv_file, nrows=0, sep=';').columns
        types_dict.update({col: np.float64 for col in col_names if col not in types_dict})

        self.signatures_frame = pd.read_csv(csv_file, sep=';', dtype=types_dict)
        num_sites_init = len(self.signatures_frame)
        self.signatures_frame.drop('slope_fdc', axis=1, inplace=True)
        self.signatures_frame.dropna(inplace=True)

        extra_sigs = [r'C:\hydro\extra_sigs\gw_array.csv', r'C:\hydro\extra_sigs\of_array2.csv']
        self.extra_sigs_files = CamelsDataset.read_attributes(extra_sigs, ',')
        CamelsDataset.remove_nan(self.extra_sigs_files)

        self.root_dir_climate = root_dir_climate
        self.root_dir_flow = root_dir_flow
        self.years_per_sample = num_years

        """number of samples depends on years/sample. """
        num_sites = len(self.gauge_id_file)
        print(f"Dropped {num_sites_init-num_sites} of {num_sites_init} sites because of nan")

        for name, normalizer in dataset_properties.sig_normalizers.items():
            self.signatures_frame[name] = self.signatures_frame[name].transform(lambda x: x * normalizer)

        """Check amount of flow data for each site and build a table of this"""
        self.siteyears = pd.DataFrame(index=self.signatures_frame.iloc[:, 0],
                                      columns=['MinYear', 'MaxYear', 'NumYears', 'NumSamples', 'RunningTotal',
                                               'Flowmean_mmd', 'flow_std',
                                               "dayl_av", "prcp_av", "srad_av",
                                               "swe_av",
                                               "tmax_av", "tmin_av", "vp_av",
                                               "dayl_std", "prcp_std", "srad_std",
                                               "swe_std",
                                               "tmax_std", "tmin_std", "vp_std"])
        self.num_samples = 0
        self.all_items = []

        if subsample_data > 0:
            num_to_load = max(int(num_sites / subsample_data), 1)
            for idx_site in range(num_to_load):
                print(f"Load {idx_site}/{num_to_load}")
                self.load_one_site(dataset_properties, idx_site)
        else:
            while len(self.all_items) == 0:
                idx_site = np.random.randint(0, num_sites) if gauge_id is None else (self.gauge_id_file.loc[self.gauge_id_file['gauge_id'] == gauge_id].index[0])
                self.load_one_site(dataset_properties, idx_site, ablation_train, ablation_validate)

    @staticmethod
    def remove_nan(df):
        for name in df.columns:
            if name != 'gauge_id':
                if np.isnan(df[name]).any():
                    median = np.nanmedian(df[name])
                    df[name][np.isnan(df[name])] = median
                    print("Replacing nan with median=" + str(median) + " in " + name)
                if np.isinf(df[name]).any():
                    max_finite = np.nanmax(df[name][np.isfinite(df[name])]) + 1
                    df[name][np.isinf(df[name])] = max_finite
                    print("Replacing inf with max + 1=" + str(max_finite) + " in " + name)

    @staticmethod
    def read_attributes(csv_file_attrib, sep):
        combined_attrib_file=None
        for file in csv_file_attrib:
            attrib_file = pd.read_csv(file, sep=sep, dtype={'gauge_id': str})
            print('Loaded columns ' + str(attrib_file.columns) + ' from ' + file)
            # Could do some labels-to-1-hot
            if combined_attrib_file is None:
                combined_attrib_file = attrib_file
            else:
                combined_attrib_file = pd.merge(left=combined_attrib_file, right=attrib_file, left_on='gauge_id',
                                             right_on='gauge_id')
        return combined_attrib_file

    def load_one_site(self, dataset_properties, idx_site, ablation_train=False, ablation_validate=False):
        """Read in climate and flow data for this site"""
        gauge_id = str(self.gauge_id_file.iloc[idx_site, 0])
        flow_data_name = self.get_streamflow_filename(gauge_id)
        flow_data = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                names=["year", "month", "day", "flow(cfs)", "qc"])
        #sigs = self.signatures_frame.loc[self.signatures_frame['gauge_id'] == gauge_id]
        climate_data_name = self.get_met_filename(gauge_id)
        climate_data = pd.read_csv(climate_data_name, sep='\t', skiprows=4, header=None,
                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                   parse_dates=True,
                                   names=["date"] + list(dataset_properties.climate_norm.keys()))
        climate_data['date'] = pd.to_datetime(climate_data['date'], format='%Y %m %d %H')  # - pd.Timedelta('12 hours')
        climate_data['date'] = climate_data['date'].apply(lambda x: x.date())
        if len(dataset_properties.climate_norm) + 1 != len(climate_data.columns):
            raise Exception("Length of DatasetProperties.climate_norm must match number of columns loaded, "
                            "excluding date (TODO add code to remove unwanted columns if needed)")
        for name, normalizer in dataset_properties.climate_norm.items():
            climate_data[name] = climate_data[name].transform(lambda x: x * normalizer)
        """Missing data label converted to 0/1 TODO how often is this happening? """
        d = {'A': 0, 'A:e': 0, 'M': 1}
        flow_data["qc"] = flow_data["qc"].map(d)
        flow_data["qc"][np.isnan(flow_data["qc"])] = 1
        flow_data["qc"] = flow_data["qc"].cumsum()  # accumulate
        # Iterate over water years
        water_year_month = 10
        water_year_day = 1
        select_water_year = (flow_data["month"] == water_year_month) & (flow_data["day"] == water_year_day)
        water_years = flow_data[select_water_year]
        indices = flow_data.index[select_water_year]

        range_end = water_years.shape[0] - self.years_per_sample
        if ablation_train:
            range_end = range_end - 3
        range_start = (range_end - 3) if ablation_validate else 0

        if range_start >= range_end:
            print("Insufficient data for site " + gauge_id)
            return

        for year_idx in range(range_start, range_end):
            record_start = water_years.iloc[year_idx]
            record_end = water_years.iloc[year_idx + self.years_per_sample]
            if record_end['qc'] > record_start['qc']:
                continue  # There's bad data in this interval

            flow_data_subset = flow_data.iloc[list(range(indices[year_idx], indices[year_idx +
                self.years_per_sample]))].reset_index(drop=True)
            # print(flow_data_subset.columns)
            flow_date_start = pd.datetime(int(record_start['year']), int(record_start['month']),
                                          int(record_start['day'])).date()
            flow_date_end = pd.datetime(int(record_end['year']), int(record_end['month']),
                                        int(record_end['day'])).date()

            climate_data_subset = climate_data.loc[(climate_data['date'] >= flow_date_start) &
                                                   (climate_data['date'] < flow_date_end)].reset_index(drop=True)

            if climate_data_subset.shape[0] != flow_data_subset.shape[0]:
                print("Missing climate data")
                continue

            if climate_data_subset.date[0] != flow_date_start:
                raise Exception("Flow data start doesn't match climate data start")

            self.clamp_length(dataset_properties, flow_data_subset)
            self.clamp_length(dataset_properties, climate_data_subset)

            self.all_items.append(self.load_hyddata(gauge_id, dataset_properties, flow_date_start,
                                                    flow_data_subset.drop(['year', 'month', 'day'], axis=1),
                                                    climate_data_subset.drop(['date'], axis=1)))

    def clamp_length(self, dataset_properties, flow_data_subset):
        length_days = 365*self.years_per_sample
        if flow_data_subset.shape[0] > length_days:
            flow_data_subset.drop(
                flow_data_subset.tail(flow_data_subset.shape[0] - length_days).index, inplace=True)

    def get_subdir_filename(self, root_dir, gauge_id, file_suffix):
        #gauge_id = gauge_id_nozero if len(gauge_id_nozero) == 8 else ('0'+str(gauge_id_nozero))
        flow_file = gauge_id + file_suffix
        subdirs = os.listdir(root_dir)
        for dirname in subdirs:
            flow_data_name = os.path.join(root_dir, dirname, flow_file)
            if os.path.exists(flow_data_name):
                return flow_data_name
        raise Exception(flow_file + ' not found in ' + self.root_dir_flow)

    def get_streamflow_filename(self, gauge_id):
        return self.get_subdir_filename(self.root_dir_flow, gauge_id, '_streamflow_qc.txt')

    def get_met_filename(self, gauge_id):
        return self.get_subdir_filename(self.root_dir_climate, gauge_id, '_lump_cida_forcing_leap.txt')

    def __len__(self):

        return len(self.all_items)

    def __getitem__(self, idx):
        if len(self.all_items) > 0:
            return self.all_items[idx]
        else:
            return self.load_item(idx)

    def check_dataframe(self, df):
        if df.isnull().any().any() or df.isin([-999, np.NAN, np.inf]).any().any():
            raise Exception('Bad input: nan/inf/null/-999')

    """
    def load_item(self, idx):
        print('load ', idx, '/', self.num_samples)
        "" "Allow for each site corresponding to multiple samples" ""
        idx_site = self.siteyears.index.get_loc(self.siteyears.index[self.siteyears['RunningTotal'] > idx][0])
        if idx_site == 0:
            idx_within_site = idx
        else:
            idx_within_site = idx - self.siteyears.iloc[idx_site - 1, 4]
        #  print("idx = ", idx, "idx_site = ", idx_site, ", idx_within_site = ", idx_within_site)

        "" "Get file names for climate and flow" ""
        gauge_id = str(self.signatures_frame.iloc[idx_site, 0])
        climate_data, flow_data, flow_data_ymd = self.load_flow_climate_csv(gauge_id)
        raise Exception("Sort out water year dates")
        minyeartmp = flow_data_ymd[(flow_data_ymd["month"] == 1) & (flow_data_ymd["day"] == 1)].min(axis=0)["year"]
        minyearidx = minyeartmp + idx_within_site * self.years_per_sample
        #  Find years for flow data
        flow_date_start = pd.datetime(minyearidx, 1, 1)
        flow_date_end = flow_date_start + pd.Timedelta('729 days')

        flow_data = flow_data.loc[(flow_data['date'] >= flow_date_start) &
                                  (flow_data['date'] <= flow_date_end)]
        flow_data = flow_data.reset_index(drop=True)

        #  print("Extracted Flow Data")

        " ""Normalize flow data" ""
        " ""First to mm/d" ""

        " ""if self.normalize_inputs:
            ""Then normalize""
            flow_data["flow(cfs)"] = ((flow_data["flow(cfs)"] - self.flow_norm.iloc[0, 0])/self.flow_norm.iloc[0, 1])
            " ""

        climate_data = climate_data.loc[(climate_data['date'] >= flow_date_start) &
                                        (climate_data['date'] <= flow_date_end)]
        climate_data = climate_data.reset_index(drop=True)

        print("Av flow=" + str(np.mean(flow_data["flow(cfs)"])))
        print("Av rain=" + str(np.mean(climate_data["prcp(mm/day)"])))

        #self.check_dataframe(flow_data)
        #self.check_dataframe(climate_data)

        " ""Merge climate and flow into one array" "" #TODO do this in the datapoint instead, only for encoder
        #hyd_data = pd.concat([flow_data.drop('date', axis=1), climate_data.drop(['date'  #, 'swe(mm)'
        #                                                                         ], axis=1)], axis=1, join='inner')
        self.check_dataframe(flow_data)
        self.check_dataframe(climate_data)

        return self.load_hyddata(gauge_id, flow_date_start, flow_data, climate_data)"""

    def load_hyddata(self, gauge_id, dataset_properties, flow_date_start, flow_data, climate_data):
        attribs = self.attrib_files.loc[self.attrib_files['gauge_id'] == gauge_id].drop('gauge_id', axis=1)
        flow_area = attribs['area_gages2']/dataset_properties.attrib_normalizers['area_gages2']
        flow_data["flow(cfs)"] = flow_data["flow(cfs)"] * cfs2mm / float(flow_area)

        flow_data = flow_data.drop('qc', axis=1)

        CamelsDataset.check_unit_size(attribs)

        signatures = self.signatures_frame.loc[self.signatures_frame['gauge_id'] == gauge_id].drop('gauge_id', axis=1)
        CamelsDataset.check_unit_size(signatures)

        self.check_dataframe(signatures)

        extra_signatures = self.extra_sigs_files.loc[self.extra_sigs_files['gauge_id'] == str(int(gauge_id))].drop('gauge_id', axis=1)
        CamelsDataset.check_unit_size(extra_signatures)
        self.check_dataframe(extra_signatures)

        latlong = self.latlong.loc[self.latlong['gauge_id'] == gauge_id]
        CamelsDataset.check_unit_size(latlong)

        return DataPoint(gauge_id+'-'+str(flow_date_start), torch.tensor(flow_data.values), flow_data.columns.tolist(),
                         torch.tensor(climate_data.values), climate_data.columns.tolist(), signatures, extra_signatures,
                         attribs, latlong)

    @staticmethod
    def check_unit_size(attribs):
        if len(attribs) != 1:
            raise Exception(f"Failed to load exactly one record. Got {len(attribs)}")

    def load_flow_climate_csv(self, gauge_id):

        if gauge_id in self.all_csv_files.keys():
            return self.all_csv_files[gauge_id]

        climate_file = self.get_met_filename(gauge_id)
        flow_data_name = self.get_streamflow_filename(self.signatures_frame.loc[gauge_id])
        climate_data_name = os.path.join(self.root_dir_climate, climate_file)
        #  print("Got file names")
        """Extract correct years out of each file"""
        flow_data_ymd = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                    names=["year", "month", "day", "flow(cfs)", "qc"])
        flow_data = pd.read_csv(flow_data_name, sep='\s+', header=None, usecols=[1, 2, 3, 4, 5],
                                parse_dates=[[1, 2, 3]])
        flow_data.columns = ["date", "flow(cfs)", "qc"]
        # convert to float
        flow_data["flow(cfs)"] = flow_data["flow(cfs)"].astype(np.float64)
        """Missing data label converted to 0/1"""
        d = {'A': 0, 'A:e': 0, 'M': 1}
        flow_data["qc"] = flow_data["qc"].map(d)

        climate_data = pd.read_csv(climate_data_name, sep='\t', skiprows=4, header=None,
                                   usecols=[0, 1, 2, 3, 4,
                                            5, 6, 7],
                                   parse_dates=True,
                                   names=["date", "dayl(s)", "prcp(mm/day)", "srad(W/m2)",
                                          "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"])
        climate_data['date'] = pd.to_datetime(climate_data['date'], format='%Y %m %d %H') - pd.Timedelta('12 hours')
        #  print("Extracted Climate Data")
        """Normalize climate data"""
        # These have high means
        for name, normalizer in self.climate_norm.items():
            climate_data[name] = climate_data[name].transform(lambda x: x * normalizer)

        self.all_csv_files[gauge_id] = [climate_data, flow_data, flow_data_ymd]
        return self.all_csv_files[gauge_id]


class ToTensor(object):
    """Convert pandas data frames in sample to Tensors."""

    def __call__(self, sample):
        gauge_id, date_start, hyd_data, signatures = sample['gauge_id'], sample['date_start'], sample['hyd_data'], \
                                                     sample['signatures']  # , sample['hyd_data_labels']
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
        return [gauge_id, date_start, hyd_data_tensor, signatures_tensor]  #  , list(hyd_data_labels)]
