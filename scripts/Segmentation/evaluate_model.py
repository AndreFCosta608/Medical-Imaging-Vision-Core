import os
import cv2
import torch
import numpy as np
from PIL import Image
import scripts.config as config
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import scripts.Segmentation.augment as augment
from scripts.Segmentation.models import ResNetUNet

def run():
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modelo = "/best_model.pt"
    model = ResNetUNet(num_classes=2)
    model.load_state_dict(torch.load(config.checkpoints + modelo, map_location=config.device))
    
    #modelo = "/modelo_completo.pth"
    #model = torch.load(config.checkpoints + modelo, map_location=config.device) #full model
    
    #modelo = "/checkpoint_epoch_20.pt"
    #checkpoint = torch.load(config.checkpoints + modelo, map_location=config.device)
    #model = ResNetUNet(num_classes=2)
    #model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(config.device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((config.height, config.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    
    eval_dir = config.extraTests
    image_files = [f for f in os.listdir(eval_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images for evaluation.")
    
    for file_name in image_files:
        img_path = os.path.join(eval_dir, file_name)
        image = Image.open(img_path).convert("L")  # Grayscale
        input_tensor = transform(image).unsqueeze(0).to(config.device)  # shape: [1, 1, H, W]
    
        #with torch.no_grad():
        if config.USE_TTA:
            output = augment.predict_with_tta(model, input_tensor)
        else:
            output = model(input_tensor)
            
        output = torch.softmax(output, dim=1)
        output = torch.argmax(output, dim=1).squeeze(0)
    
        if config.USE_REFINEMENT:
            output = augment.refine_mask(output)
    
        predicted_mask = output.cpu().numpy()  # shape [H, W]
        image_np = np.array(image.resize((config.width, config.height)), dtype=np.float32) / 255.0  # [H, W]
    
        mask_overlay = np.zeros((config.height, config.width, 3), dtype=np.float32)
        mask_overlay[..., 0] = predicted_mask  # vermelho onde a máscara = 1
    
        image_rgb = np.stack([image_np]*3, axis=-1)
    
        alpha = 0.4
        blended = (1 - alpha) * image_rgb + alpha * mask_overlay
    
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].imshow(image_np, cmap='gray')
        axs[0].set_title('Rated Image')
        axs[0].axis('off')
    
        axs[1].imshow(predicted_mask, cmap='jet')
        axs[1].set_title('Predicted Mask')
        axs[1].axis('off')
    
        axs[2].imshow(blended)
        axs[2].set_title('Overlay')
        axs[2].axis('off')
    
        plt.suptitle(f"Rating: {file_name}", fontsize=12)
        plt.tight_layout()
        plt.show()
        
        input("Press ENTER to continue...")
    
    print('\n\nCompleted...')
    
if __name__ == "__main__":
    run()
