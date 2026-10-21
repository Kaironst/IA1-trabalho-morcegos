import torch
from torch.utils.data import Dataset


class SpeciesDataset(Dataset):
    def __init__(self, data_path, name_path, is_training):
        self.is_training = is_training
        with open(data_path, "r") as f:
            self.data = f.readlines()
        with open(name_path, "r") as f:
            self.names = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = [float(val) for val in self.data[idx].split()]
        species_name = (self.names[idx])[:81]

        features_tensor = torch.tensor(features)

        if self.is_training:
            is_bat = int((self.names[idx])[81])
            class_tensor = torch.tensor(is_bat)
            return features_tensor, class_tensor, species_name
        else:
            return features_tensor, species_name
