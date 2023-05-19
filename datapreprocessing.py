# unzip file
# !unzip /content/dbnFiles.zip

# download if you need
# !pip install ViennaRNA


import time
from utils import get_onehot, original_dataset, calculate_energy, MyDataset
import torch

# dbn file
start_time = time.time()
dbnFile = '/content/dbnFiles'
dbn_onehot = get_onehot(dbnFile)
print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
# 102128

# get original dataset
start_time = time.time()
dbnFile_original = original_dataset(dbnFile)
print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

# calculate energy
start_time = time.time()
mfe_dbn = []
for each in dbnFile_original:
    sequence = each[0]
    structure = each[1]
    energy = calculate_energy(sequence, structure)
    mfe_dbn.append(energy)
print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))


train_data = MyDataset(dbn_onehot, mfe_dbn)

torch.save(train_data, "train_dataV2.pt")