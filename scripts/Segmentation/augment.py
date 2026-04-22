import cv2
import torch
import numpy as np

def predict_with_tta(model, image):
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.rot90(x, 1, [2, 3])
    ]
    predictions = []
    for tf in transforms:
        aug = tf(image)
        #with torch.no_grad():
        pred = model(aug)
        inv_pred = tf(pred)
        #predictions.append(torch.softmax(inv_pred, dim=1))
        predictions.append(inv_pred)
    
    #avg_pred = torch.stack(predictions).mean(0)
    #return torch.argmax(avg_pred, dim=1).squeeze(0)
    avg_logits = torch.stack(predictions).mean(0)  # [B, C, H, W]
    return avg_logits

def refine_mask(mask_tensor):
    mask = mask_tensor.cpu().numpy().astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy(opened).to(mask_tensor.device)
