import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from diffusers import UNet2DConditionModel, DDIMScheduler
from tqdm import tqdm
from train_diffusion import PhysicsProjector

def generar_galaxia(modelo_path, masa_norm, sfr_norm, radio_norm, img_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu':
        torch.set_num_threads(os.cpu_count() or 4)
        
    print(f"🌌 Iniciando generación en: {device} (Hilos: {torch.get_num_threads()})")

    unet = UNet2DConditionModel(
        sample_size=img_size,
        in_channels=3,
        out_channels=3,
        cross_attention_dim=256,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512), 
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    
    # IMPORTANTE: input_dim=3 porque el nuevo cerebro aprendió con Masa, SFR y Radio
    projector = PhysicsProjector(input_dim=3).to(device)

    # Cargar pesos (añadimos weights_only=True para quitar el warning rojo)
    checkpoint = torch.load(modelo_path, map_location=device, weights_only=True)
    unet.load_state_dict(checkpoint['unet'])
    projector.load_state_dict(checkpoint['projector'])
    
    unet.eval()
    projector.eval()

    scheduler = DDIMScheduler(num_train_timesteps=1000)

    image = torch.randn((1, 3, img_size, img_size)).to(device)
    
    # Creamos el tensor con los 3 parámetros físicos que has elegido
    phys_vector = torch.tensor([[masa_norm, sfr_norm, radio_norm]], dtype=torch.float32).to(device)
    
    PASOS_INFERENCIA = 50 
    scheduler.set_timesteps(PASOS_INFERENCIA)
    
    with torch.no_grad():
        encoder_hidden_states = projector(phys_vector)
        
        for t in tqdm(scheduler.timesteps, desc=f"Esculpiendo galaxia ({PASOS_INFERENCIA} pasos)"):
            noise_pred = unet(image, t, encoder_hidden_states=encoder_hidden_states).sample
            image = scheduler.step(noise_pred, t, image).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1) 
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Galaxia Generada\nMasa={masa_norm:.2f} | SFR={sfr_norm:.2f} | Radio={radio_norm:.2f}")
    
    # Guardamos la imagen en alta calidad para tu TFG
    plt.savefig("mi_tercera_galaxia.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    RUTA_NUEVO_MODELO = "checkpoints/modelo_epoca_10.pt" 
    
    generar_galaxia(
        modelo_path=RUTA_NUEVO_MODELO,
        masa_norm=0.8,   # Galaxia masiva
        sfr_norm=0.9,    # Mucha formación estelar (brazos azules)
        radio_norm=0.5,  # Tamaño medio
        img_size=512
    )