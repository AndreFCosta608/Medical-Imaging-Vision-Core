---
license: mit
tags:
  - image-segmentation
  - unet
  - resnet
  - computer-vision
  - pytorch
library_name: transformers
datasets:
  - antoreepjana/cv-image-segmentation
inference: false
---

# ? Model Card ? Segmentation


## 🧾 Overview

💡 **ResNet + U-Net fusion** combines deep and contextual vision (ResNet) with spatial fidelity and precision in details (U-Net).  
It is a versatile, powerful, and high-sensitivity architecture — ideal for projects where **every pixel matters**.

The model excels in scenarios where the object is **small, detailed, or textured**, and where the **global scene context offers little help**.

This makes it ideal for:
- Medical segmentation (e.g., tumors, vessels)
- Industrial defect inspection
- Embedded vision for robotics or precision tasks

⚠️ However, this specific version was trained on a **narrow-domain dataset**, captured under **controlled indoor conditions**: consistent lighting, high-contrast backgrounds, and fixed camera angles.  
As a result, its ability to generalize to open-world scenarios (e.g., outdoor environments, variable backgrounds) is limited.  

**This is not a flaw in the model**, but a **natural reflection of the training data**.  
When retrained with more diverse and realistic datasets, this architecture is highly capable of delivering robust performance across a wide range of segmentation tasks.

---

## ☕ Behind the Scenes

This certification project was built one commit at a time — powered by curiosity, long debugging sessions, strategic doses of caffeine, and great support from **Microsoft Copilot** and **ChatGPT (OpenAI)**, whose insights were essential in structuring the segmentation pipeline and planning its embedded future.

> "Every time the model tries to segment, the square figure resurfaces. Not as an error, but as a reminder: deep learning can be quite shallow when the curse of imperfect geometry sets in.  
> And even when all the code is rewritten, the world is realigned, and optimism rises again… there she is: the misshapen quadratic figure.  
> Unfazed, unshakeable, perhaps even moved by her own stubbornness. She's not a bug — she's a character."

---

## 🗂️ Dataset

This model was trained using a subset of the [CV Image Segmentation Dataset](https://www.kaggle.com/datasets/antoreepjana/cv-image-segmentation), available on Kaggle.

- **Author**: Antoreep Jana  
- **License**: For educational and non-commercial use  
- **Content**: 300+ annotated images for binary segmentation  
- **Preprocessing**: All images resized to 512×512 and converted to grayscale

⚠️ *Only a filtered and preprocessed subset (related to car images) was used for this version.*
The Dataset presents some distinct data subsets.
I only used the images related to carvana cars (Kaggle Carvana Car Mask Segmentation). This was the dataset used to test the project ...

---

## ⚙️ Model Architecture

- **Encoder**: ResNet-50 (pretrained, adapted for 1-channel input)
- **Decoder**: U-Net with skip connections and bilinear upsampling
- **Input**: Grayscale, 512×512
- **Output**: Binary segmentation mask (background vs. object)
- **Loss**: Composite of `CrossEntropyLoss + DiceLoss`
- **Framework**: PyTorch

---

## 📊 Evaluation Metrics

- Pixel Accuracy (train/val)
- Dice Coefficient
- CrossEntropy Loss
- Class-weighted loss balancing
- *(IoU, MCC, Precision/Recall planned for future integration)*

🧪 Evaluation performed using `evaluate_model.py`

---

## ⚠️ Limitations

This model achieves excellent results when tested on **studio-like images**: consistent lighting, neutral backgrounds, and static perspectives.

However, performance decreases on **unseen outdoor scenarios** (e.g., cars on the street, parking lots) — where background noise, lighting variation, and camera angle impact results.

➡️ This **limitation is dataset-induced**, not architectural.  
When trained on more realistic data, this model generalizes well due to its high sensitivity to texture and spatial structure.

---

## 🚀 Intended Use

Best suited for applications where conditions are similar to the training set, such as:

- Quality control in automotive photography studios
- Automated documentation of vehicles in inspection booths
- Offline image processing for structured, grayscale datasets

---

## 💡 Recommendations

To deploy in open-world environments (e.g., mobile robots, outdoor cameras), it is strongly recommended to **retrain or fine-tune** the model using a **more heterogeneous dataset**.

---

## 🔬 Planned Extensions

The following experimental modules are under active development and may be integrated in future releases:

1️⃣ **Embedded Deployment Pipeline**
- Export to ONNX format with float16 precision
- C++ reimplementation targeting edge devices such as ESP32-S3 and STM32H7
- Lightweight modular training script:  
  `scripts/Segmentation/Future/train_embedded_explicit_model.py`  
  *Status: Experimental – not validated in this version*

2️⃣ **Automated Hyperparameter Optimization**
- Training script that performs automatic hyperparameter search and tuning before final training
- Designed to improve efficiency and reduce manual configuration
- Script:  
  `scripts/Segmentation/Future/cyber_train.py`  
  *Status: Experimental – not validated in this version*


---

## 🪪 Licensing

- **Code**: MIT License  
- **Dataset**: Attribution required (as per Kaggle contributor)
