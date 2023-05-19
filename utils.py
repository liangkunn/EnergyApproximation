import os
import glob
import torch
import RNA
import time
import torch
import torch
import numpy as np
from io import open
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# one-hot for 1st sequence and 2nd structure
class OneHot(object):
    def __init__(self):
        #one-hot
        self.seq_dict = { # modified from e2efold 
            'A':np.array([1,0,0,0,0,0,0,0,0]),
            'U':np.array([0,1,0,0,0,0,0,0,0]),
            'C':np.array([0,0,1,0,0,0,0,0,0]),
            'G':np.array([0,0,0,1,0,0,0,0,0]),
            'N':np.array([0,0,0,0,0,0,0,0,0]),
            'M':np.array([1,0,1,0,0,0,0,0,0]),
            'Y':np.array([0,1,1,0,0,0,0,0,0]),
            'W':np.array([1,0,0,0,0,0,0,0,0]),
            'V':np.array([1,0,1,1,0,0,0,0,0]),
            'K':np.array([0,1,0,1,0,0,0,0,0]),
            'R':np.array([1,0,0,1,0,0,0,0,0]),
            'I':np.array([0,0,0,0,0,0,0,0,0]),
            'X':np.array([0,0,0,0,0,0,0,0,0]),
            'S':np.array([0,0,1,1,0,0,0,0,0]),
            'D':np.array([1,1,0,1,0,0,0,0,0]),
            'P':np.array([0,0,0,0,0,0,0,0,0]),
            'B':np.array([0,1,1,1,0,0,0,0,0]),
            'H':np.array([1,1,1,0,0,0,0,0,0]),
            'a':np.array([1,0,0,0,0,0,0,0,0]), # self-made
            'u':np.array([0,1,0,0,0,0,0,0,0]),
            'c':np.array([0,0,1,0,0,0,0,0,0]),
            'g':np.array([0,0,0,1,0,0,0,0,0]),
            'n':np.array([0,0,0,0,0,0,0,0,0]),
            'y':np.array([0,1,1,0,0,0,0,0,0]),
            'w':np.array([1,0,0,0,0,0,0,0,0]),
            'v':np.array([1,0,1,1,0,0,0,0,0]),
            'k':np.array([0,1,0,1,0,0,0,0,0]),
            'r':np.array([1,0,0,1,0,0,0,0,0]),
            'i':np.array([0,0,0,0,0,0,0,0,0]),
            'x':np.array([0,0,0,0,0,0,0,0,0]),
            's':np.array([0,0,1,1,0,0,0,0,0]),
            'd':np.array([1,1,0,1,0,0,0,0,0]),
            'p':np.array([0,0,0,0,0,0,0,0,0]),
            'b':np.array([0,1,1,1,0,0,0,0,0]),
            'h':np.array([1,1,1,0,0,0,0,0,0])
        }
        self.stru_dict = {
            '.': np.array([1,0,0,0,0,0,0,0,0]), 
            '(': np.array([0,1,0,0,0,0,0,0,0]), 
            ')': np.array([0,0,1,0,0,0,0,0,0]),
            '[': np.array([0,0,0,1,0,0,0,0,0]),
            ']': np.array([0,0,0,0,1,0,0,0,0]),
            '{': np.array([0,0,0,0,0,1,0,0,0]),
            '}': np.array([0,0,0,0,0,0,1,0,0]),
            '<': np.array([0,0,0,0,0,0,0,1,0]),
            '>': np.array([0,0,0,0,0,0,0,0,1]),
        }



# reading db files        
class Corpus(object):

    def __init__(self, path):
        self.dictionary = OneHot()
        self.data = self.tokenize(path)
            
    def tokenize(self, path):
        """Tokenizes a text file."""
        stfiles = glob.glob(path)

        ids1_total = []
        ids2_total = []
        for stfile in stfiles:

            with open(stfile, 'r', encoding="utf8") as f:
                ids1 = []
                ids2 = []

                for i in range(20):  # read the first 20 lines, not so important, because after reading the structure, it will break
                    line = f.readline().strip('\n')  # remove \n
                    if line.startswith("#"):  # skip the comment line
                        continue
                    elif line.startswith(tuple(self.dictionary.seq_dict)):  # Sequence
                        chars = list(line)
                        valid_file = True

                        for char in chars:
                            if char in self.dictionary.seq_dict:
                                ids1.append(self.dictionary.seq_dict[char])
                            else:
                                # if strange key appears, skip this file
                                valid_file = False
                                break
                        if valid_file:
                            ids1_total.append(torch.tensor(ids1).type(torch.int64))
                        else:
                            break
                    elif line.startswith(tuple(self.dictionary.stru_dict)):  # Secondary Structure - dot-bracket
                        chars = list(line)
                        valid_file = True
                        for char in chars:
                            if char in self.dictionary.stru_dict:
                                ids2.append(self.dictionary.stru_dict[char])
                            else:
                                valid_file = False
                                break
                        if valid_file:
                            ids2_total.append(torch.tensor(ids2).type(torch.int64))
                        break
                    else:
                        print(stfile)

        return ids1_total, ids2_total
    

def get_onehot(directory_path):
    corpus_data = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".db") or filename.endswith(".dbn"):
            # get file path
            filepath = os.path.join(directory_path, filename)
            # use corpus
            corpus = Corpus(filepath)
            data = corpus.tokenize(filepath)

            # check if data has at least one element in both lists, if not -> skip
            if data[0] and data[1]:
                # concatenated value will be used as input for network
                data_concat = torch.cat([data[0][0], data[1][0]], dim=1)
                corpus_data.append(data_concat)
            
    return corpus_data

# use the Vienna RNA package to calculate experimental energy
def calculate_energy(sequence, structure):
    params = RNA.fold_compound(sequence)
    energy = params.eval_structure(structure)
    return energy

# save original dataset
def original_dataset(directory_path):
    concatenated_data = []
    
    valid_seq_chars = set("ACGUNMYWVKRIXSDPBHacgunywvkrixsdpbh")
    valid_stru_chars = set(".()[]{}<>")
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".db") or filename.endswith(".dbn"):
            filepath = os.path.join(directory_path, filename)
            
            with open(filepath, "r") as f:
                lines = f.readlines()

            rna_sequence = None
            secondary_structure = None
            for line in lines:
                if line.startswith("#"):
                    continue
                elif rna_sequence is None:
                    rna_sequence = line.strip()
                elif secondary_structure is None:
                    secondary_structure = line.strip()
                    break
            
            if all(char in valid_seq_chars for char in rna_sequence) and all(char in valid_stru_chars for char in secondary_structure):
                concatenated_data.append([rna_sequence, secondary_structure])
            
    return concatenated_data

# custom dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {"input": self.data[idx], "label": self.labels[idx]}
        return sample
        