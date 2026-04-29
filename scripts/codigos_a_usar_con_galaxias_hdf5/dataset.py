import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import numpy as np
import json
from PIL import Image

class GalaxiasFisicasDataset(Dataset):
    def __init__(self, hdf5_path, img_size):
        self.hdf5_path = hdf5_path
        self.img_size = img_size
        self.h5_file = None # Se abrirá dinámicamente para los workers
        
        # Leemos el índice y las stats de golpe y cerramos para no bloquear
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = len(f['images'])
            self.stats = json.loads(f.attrs['stats'])
            
        print(f"⚡ Dataset HDF5 enganchado: {self.length} galaxias listas para procesar.")
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return self.length

    def _normalize(self, value, name):
        v_min = self.stats[name]['min']
        v_max = self.stats[name]['max']
        return (value - v_min) / (v_max - v_min + 1e-8)

    def __getitem__(self, idx):
        # Cada "hilo" de PyTorch abre el archivo independientemente = Velocidad Máxima
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            
        # 1. Leer imagen y transformarla
        img_np = self.h5_file['images'][idx]
        image = Image.fromarray(img_np)
        image = self.transform(image)
        
        # 2. Físicas con Aumento de Datos (ruido gaussiano basado en su error)
        phys_data = self.h5_file['fisica'][idx]
        v_mass, err_mass = phys_data[0], phys_data[1]
        v_sfr, err_sfr   = phys_data[2], phys_data[3]
        v_rad, err_rad   = phys_data[4], phys_data[5]
        
        val_mass = np.random.normal(loc=v_mass, scale=err_mass)
        val_sfr  = np.random.normal(loc=v_sfr, scale=err_sfr)
        val_radio = np.random.normal(loc=v_rad, scale=err_rad)
        
        norm_mass  = self._normalize(val_mass, 'LOG_MS')
        norm_sfr   = self._normalize(val_sfr, 'SFR')
        norm_radio = self._normalize(val_radio, 'RADIO_P')
        
        fisica_vector = torch.tensor([norm_mass, norm_sfr, norm_radio], dtype=torch.float32)
        fisica_vector = torch.clamp(fisica_vector, 0.0, 1.0)
        
        # 3. Leer secuencia RGB
        rgb_np = self.h5_file['rgb'][idx]
        rgb_vector = torch.tensor(rgb_np, dtype=torch.float32)
        
        return image, fisica_vector, rgb_vector