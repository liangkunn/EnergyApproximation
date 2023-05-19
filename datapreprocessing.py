# unzip file
# !unzip /content/dbnFiles.zip

# download if you need
# !pip install ViennaRNA


import time
from utils import get_onehot, original_dataset, calculate_energy, MyDataset, filter_by_length, filter_by_energy, find_extremes, custom_collate
import torch
import random
from torch.utils.data import SubsetRandomSampler, Dataset, DataLoader, random_split

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

# make our own dataset
train_data = MyDataset(dbn_onehot, mfe_dbn)

# save data
torch.save(train_data, "train_dataV2.pt")

# Load the data from the file using torch.load()
loaded_data = torch.load("/content/train_dataV2.pt")
len(loaded_data)

ml, maxe, mine = find_extremes(loaded_data)
print("The maximum length of original data is:", ml)
print("The maximum energy of original data is:", maxe)
print("The minimum energy of original data is:", mine)

# filter data
filtered_length = filter_by_length(loaded_data, 512) # 800 for transformer, 512 for DistilBert
filtered_length_dataset = MyDataset(filtered_length['input'], filtered_length['label'])
print("filtered data amount:", len(filtered_length_dataset))

# filter extreme energy values
filtered_energy = filter_by_energy(filtered_length_dataset, max_energy=50, min_energy=-50)

# dataset is prepared
loaded_dataset = MyDataset(filtered_energy['input'], filtered_energy['label'])

random_seed = 901
random.seed(random_seed)  # set the random seed for reproducibility

# Create the dataset
inputs = [item['input'] for item in loaded_dataset]
labels = [item['label'] for item in loaded_dataset]

# the whole dataset as follow:
all_data = MyDataset(inputs, labels)
all_data_indices = list(range(len(all_data)))  # create a list of indices

# train subsampler
train_samples = 6000 # 9000 on transformerV2 beta3
train_random_indices = random.sample(all_data_indices, train_samples)  # randomly select indices
train_random_sampler = SubsetRandomSampler(train_random_indices)

# val subsampler
val_samples = 1500 # 3000 on transformerV2 beta3
# remove the indices of the train samples from the list of available indices
available_indices = list(set(all_data_indices) - set(train_random_indices))
val_random_indices = random.sample(available_indices, val_samples)  # randomly select indices
val_random_sampler = SubsetRandomSampler(val_random_indices)

# test subsampler
test_samples = 1500 # 3000 on transformerV2 beta3
# remove the indices of the validation samples from the list of available indices
available_indices = list(set(available_indices) - set(train_random_indices) - set(val_random_indices))
test_random_indices = random.sample(available_indices, test_samples)  # randomly select indices
test_random_sampler = SubsetRandomSampler(test_random_indices)

# batch_size
batch_size=16 # save memory, otherwise it we wasily get 'out of memory error'

# Create a PyTorch dataloader using the data
train_loader = DataLoader(all_data, batch_size=batch_size, sampler=train_random_sampler, collate_fn=custom_collate)
val_loader = DataLoader(all_data, batch_size=batch_size, sampler=val_random_sampler, collate_fn=custom_collate)
test_loader = DataLoader(all_data, batch_size=batch_size, sampler=test_random_sampler, collate_fn=custom_collate)