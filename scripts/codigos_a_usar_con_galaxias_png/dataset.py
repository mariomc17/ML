import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import json # Importamos json al principio

class GalaxiasFisicasDataset(Dataset):
    def __init__(self, csv_path, csv_brazos_path, images_dir, img_size):
        self.images_dir = images_dir
        
        # 1. Cargar AMBOS CSVs asegurando que OBJID es texto
        df_crudo = pd.read_csv(csv_path, dtype={'OBJID': str})
        df_brazos = pd.read_csv(csv_brazos_path, dtype={'OBJID': str})
        
        # 2. Fusionar los datos (MERGE): 
        # Esto junta las columnas físicas y las columnas R, G, B en una sola mega-tabla
        df_unido = pd.merge(df_crudo, df_brazos, on='OBJID', how='inner')
        
        # 3. Filtrar: Quedarnos SOLO con las filas que tienen imagen real
        galaxias_validas = []
        for idx, row in df_unido.iterrows():
            objid = str(row['OBJID'])
            ruta_imagen = os.path.join(self.images_dir, f"{objid}.png")
            if os.path.exists(ruta_imagen):
                galaxias_validas.append(row)
        
        self.df = pd.DataFrame(galaxias_validas).reset_index(drop=True)
        print(f"🚀 Dataset cargado: Se encontraron {len(self.df)} imágenes válidas con datos físicos y de brazos.")
        
        # 4. STATS para Min-Max Scaling (Solo de los 3 parámetros elegidos)
        self.stats = {
            'LOG_MS':  {'min': self.df['LOG_MS'].min(),  'max': self.df['LOG_MS'].max()},
            'SFR':     {'min': self.df['SFR'].min(),     'max': self.df['SFR'].max()},
            'RADIO_P': {'min': self.df['RADIO_P'].min(), 'max': self.df['RADIO_P'].max()},
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Rango [-1, 1]
        ])

    def __len__(self):
        return len(self.df)

    def _normalize(self, value, name):
        v_min = self.stats[name]['min']
        v_max = self.stats[name]['max']
        return (value - v_min) / (v_max - v_min + 1e-8)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        objid = str(row['OBJID'])
        img_path = os.path.join(self.images_dir, f"{objid}.png")
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        # ================================================================
        # DATA AUGMENTATION ESTOCÁSTICO (Muestreo Gaussiano)
        # ================================================================

        err_mass  = max(0.0, float(row['LOG_MS_ERR']))
        err_sfr   = max(0.0, float(row['SFR_ERR']))
        err_radio = max(0.0, float(row['RADIO_P_ERR']))
        
        val_mass = np.random.normal(loc=row['LOG_MS'], scale=err_mass)
        val_sfr = np.random.normal(loc=row['SFR'], scale=err_sfr)
        val_radio = np.random.normal(loc=row['RADIO_P'], scale=err_radio)
        
        norm_mass  = self._normalize(val_mass, 'LOG_MS')
        norm_sfr   = self._normalize(val_sfr, 'SFR')
        norm_radio = self._normalize(val_radio, 'RADIO_P')
        
        fisica_vector = torch.tensor([norm_mass, norm_sfr, norm_radio], dtype=torch.float32)
        fisica_vector = torch.clamp(fisica_vector, 0.0, 1.0)
        
        # ================================================================
        # VECTORES DE LA ESTRUCTURA ESPIRAL (RGB) DESDE EL CSV
        # ================================================================
        r_arr = json.loads(row['R_array'])
        g_arr = json.loads(row['G_array'])
        b_arr = json.loads(row['B_array'])
        
        rgb_vector = torch.tensor([r_arr, g_arr, b_arr], dtype=torch.float32)

        return image, fisica_vector, rgb_vector