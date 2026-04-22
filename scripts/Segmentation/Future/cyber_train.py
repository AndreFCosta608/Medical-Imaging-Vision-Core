import os
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scripts.config as config
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from torch.utils.data import random_split
import torchvision.transforms as transforms
import scripts.Segmentation.augment as augment
from scripts.Segmentation.models import ResNetUNet
from warmup_scheduler import GradualWarmupScheduler
from scripts.Segmentation.focalLoss import FocalLoss
from scripts.Segmentation.segDS import SegmentationDataset
from scripts.Segmentation.diceLossCriterion import DiceLoss

def save_report(row):
    print(str(row))
    with open(config.report_file, 'a', encoding='utf-8') as f:
        f.write(str(row) + '\n')

def compute_class_weights(dataset):
    class_counts = torch.zeros(2)
    for _, mask in dataset:
        pixels = mask.view(-1)
        for c in [0, 1]:
            class_counts[c] += (pixels == c).sum()
    weights = class_counts.sum() / (2.0 * class_counts + 1e-6)
    weights = weights / weights.sum()
    return weights

def auto_detect_hyperparams(model, train_loader, val_loader):
    candidate_configs = [
        {"lr": 1e-3, "wd": 1e-5, "use_focal": False, "dice_weight": 1.0},
        {"lr": 1e-4, "wd": 1e-4, "use_focal": True, "dice_weight": 1.5},
        {"lr": 5e-4, "wd": 1e-5, "use_focal": False, "dice_weight": 2.0},
        {"lr": 1e-4, "wd": 1e-6, "use_focal": True, "dice_weight": 1.0},
    ]
   
    best_score = -float('inf')
    best_config = candidate_configs[0]
    
    for cfg in candidate_configs:
        temp_model = ResNetUNet(num_classes=2).to(config.device)
        optimizer = optim.AdamW(temp_model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        criterion = FocalLoss(gamma=2.0) if cfg["use_focal"] else nn.CrossEntropyLoss()
        dice_loss = DiceLoss()
        temp_model.train()
        for i, (images, masks) in enumerate(train_loader):
            if i > 2: break
            images = images.to(config.device).float()
            masks = masks.to(config.device).long()
            outputs = temp_model(images)
            loss = cfg["dice_weight"] * dice_loss(outputs, masks) + 0.5 * criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        temp_model.eval()
        val_acc = 0
        val_loss = 0
        val_batches = 0
        #with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(config.device).float()
            masks = masks.to(config.device).long()
            outputs = temp_model(images)
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == masks).sum().item() / torch.numel(masks)
            val_acc += acc
            loss = cfg["dice_weight"] * dice_loss(outputs, masks) + 0.5 * criterion(outputs, masks)
            val_loss += loss.item()
            val_batches += 1
            if val_batches > 2: break
        
        avg_score = val_acc / val_batches - val_loss / val_batches
        if avg_score > best_score:
            best_score = avg_score
            best_config = cfg
    
    return best_config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
try:
    save_report(torch.__version__)
    save_report(torch.version.cuda)
    save_report(torch.cuda.get_arch_list()) #['sm_75', 'sm_86'] 
except Exception as e:
    save_report(e)
    pass

config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Starting processing...')

os.makedirs(config.checkpoints, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((config.height, config.width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = SegmentationDataset(transform = transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

model = ResNetUNet(num_classes=2).to(config.device)

class_weights = compute_class_weights(dataset).to(config.device)


selected = auto_detect_hyperparams(model, train_loader, val_loader)
save_report(f"\n🔍 Auto-selected hyperparams: {selected}")

criterion = FocalLoss(gamma=2.0) if selected["use_focal"] else nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=selected["lr"], weight_decay=selected["wd"])
dice_loss = DiceLoss()


cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=5, after_scheduler=cosine_scheduler)

dataHoraInicial = datetime.datetime.now()
save_report('\n\n\nStarting training on: ' + str(dataHoraInicial))

accuracies = []
iou_history = []
loss_history = []
dice_history = []
val_accuracies = []
best_accuracy = 0.0
dice_loss = DiceLoss()
epochs_no_improve = 0

model.train()
for epoch in range(config.num_epochs):
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0

    for images, masks in train_loader:
        images = images.to(config.device).float()
        masks = masks.to(config.device).long()
        
        output = []

        if config.USE_TTA:
            for img in images:
                preds = augment.predict_with_tta(model, img.unsqueeze(0))  # [1, C, H, W]
                output.append(preds)
            output = torch.cat(output, dim=0)  # [B, C, H, W]
        else:
            output = model(images)

        output = output.float()
        masks = masks.long()

        loss = selected["dice_weight"] * dice_loss(output, masks) + 0.5 * criterion(output, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(output, dim=1)
        
        if config.USE_REFINEMENT:
            preds = augment.refine_mask(preds)        
        
        correct_pixels += (preds == masks).sum().item()
        total_pixels += torch.numel(preds)
        total_loss += loss.item()
    
    loss_history.append(total_loss)
    epoch_accuracy = correct_pixels / total_pixels
    accuracies.append(epoch_accuracy)
    
    model.eval()
    val_correct = 0
    val_total = 0
    val_preds_all = []
    val_targets_all = []
    for val_images, val_masks in val_loader:
        val_images = val_images.to(config.device).float()
        val_masks = val_masks.to(config.device).long()
        
        val_outputs = model(val_images)
        val_preds = torch.argmax(val_outputs, dim=1)
            
        val_preds_all.append(val_preds.view(-1).cpu().numpy())
        val_targets_all.append(val_masks.view(-1).cpu().numpy())
            
        val_correct += (val_preds == val_masks).sum().item()
        val_total += torch.numel(val_preds)
            
    val_accuracy = val_correct / val_total
    val_accuracies.append(val_accuracy)
    
    val_preds_flat = np.concatenate(val_preds_all)
    val_targets_flat = np.concatenate(val_targets_all)
    
    #iou = jaccard_score(val_targets_flat, val_preds_flat, average='binary')
    iou = jaccard_score(val_targets_flat, val_preds_flat, average='macro')  # ou 'weighted'

    intersection = np.logical_and(val_preds_flat, val_targets_flat).sum()
    union = np.logical_or(val_preds_flat, val_targets_flat).sum()
    dice = (2 * intersection) / (val_preds_flat.sum() + val_targets_flat.sum() + 1e-6)
    iou_history.append(iou)
    dice_history.append(dice)

    save_report(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss:.4f}, "
          f"Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
          f"IoU: {iou:.4f}, Dice: {dice:.4f}")


    if (epoch + 1) % config.checkpoint_interval == 0:
        checkpoint_path = os.path.join(config.checkpoints, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'accuracy': epoch_accuracy,
        }, checkpoint_path)

    if epoch_accuracy > best_accuracy:
        save_report(f"🔸 New best model at epoch {epoch+1} (acc: {epoch_accuracy:.4f}) — saving best_model.pt")
        best_accuracy = epoch_accuracy
        torch.save(model.state_dict(), os.path.join(config.checkpoints, "best_model.pt"))
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= config.early_stop_patience:
        save_report(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
        break

dataHoraFinal = datetime.datetime.now()
save_report('Completing training on: ' + str(dataHoraFinal))
save_report('Total training execution time = ' + str((dataHoraFinal - dataHoraInicial)))    

model.eval()

torch.save(model, config.modelName)

try:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title("Loss Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.tight_layout()
    plt.savefig(config.source + 'training_loss.png')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, marker='o', color='green')
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Pixel Accuracy")
    plt.grid()
    plt.tight_layout()
    plt.savefig(config.source + 'training_val_accuracy.png')
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(iou_history)+1), iou_history, marker='o', color='purple')
    plt.title("IoU Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("IoU Score")
    plt.grid()
    plt.tight_layout()
    plt.savefig(config.source + 'iou_history.png')
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(dice_history)+1), dice_history, marker='o', color='orange')
    plt.title("Dice Score Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.grid()
    plt.tight_layout()
    plt.savefig(config.source + 'dice_history.png')
    
except Exception as e:
    pass    

save_report("\nTraining Summary:")
save_report(f"  Min Loss: {min(loss_history):.4f}")
save_report(f"  Max Loss: {max(loss_history):.4f}")
save_report(f"  Loss final: {loss_history[-1]:.4f}")
save_report(f"  Best Val Acc: {max(val_accuracies):.4f}")
print("\nCompleted ✅")
