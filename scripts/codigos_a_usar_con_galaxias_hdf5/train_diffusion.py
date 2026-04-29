import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.optim import AdamW
from tqdm import tqdm
import os
from dataset import GalaxiasFisicasDataset 

IMG_SIZE = 128
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# Ruta donde pondrás el archivo único .h5 en el servidor de la A100
HDF5_PATH = "/home/mario/ML/codigos_a_usar_con_galaxias_hdf5/dataset_galaxias.h5" 
CHECKPOINT_DIR = "checkpoints"
# =====================================================================

class PhysicsProjector(nn.Module):
    def __init__(self, input_dim=3, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(), 
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        projected = self.net(x)
        return projected.unsqueeze(1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento en: {device} | Imagen: {IMG_SIZE}x{IMG_SIZE}")

    dataset = GalaxiasFisicasDataset(hdf5_path=HDF5_PATH, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    unet = UNet2DConditionModel(
        sample_size=IMG_SIZE,
        in_channels=3,
        out_channels=3,
        cross_attention_dim=256, 
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512), 
        down_block_types=(
            "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D",
        ),
    ).to(device)

    projector = PhysicsProjector(input_dim=3).to(device)

    optimizer = AdamW(list(unet.parameters()) + list(projector.parameters()), lr=LR)
    criterion = nn.MSELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        unet.train()
        projector.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            clean_images, phys_vectors, rgb_vectors = batch 
            clean_images = clean_images.to(device)
            phys_vectors = phys_vectors.to(device)

            noise = torch.randn_like(clean_images)
            bsz = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            encoder_hidden_states = projector(phys_vectors)
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
        
        print(f"Época {epoch+1} terminada | MSE Loss Medio: {epoch_loss/len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'unet': unet.state_dict(),
                'projector': projector.state_dict()
            }, f"{CHECKPOINT_DIR}/modelo_epoca_{epoch+1}.pt")

if __name__ == "__main__":
    main()