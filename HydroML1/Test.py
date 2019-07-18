print('hello')

import CAMELS_data as Cd
import os

root_dir_flow = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'usgs_streamflow')
root_dir_climate = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'basin_mean_forcing', 'daymet')
root_dir_signatures = os.path.join('D:', 'Hil_ML', 'Input', 'CAMELS', 'camels_attributes_v2.0')
csv_file = os.path.join(root_dir_signatures,'camels_hydro.txt')

TestCamels = Cd.CamelsDataset(csv_file, root_dir_climate, root_dir_flow, 2)

print("Len = ", TestCamels.__len__())

testoutput = TestCamels.__getitem__(0)
print("Output = ", testoutput)
testoutput = TestCamels.__getitem__(1000)
print("Output2 = ", testoutput)
testoutput = TestCamels.__getitem__(5000)
print("Output3 = ", testoutput)
