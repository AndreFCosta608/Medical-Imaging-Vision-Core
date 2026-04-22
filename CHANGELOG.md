# 📌 Changelog

All notable changes to this project will be documented in this file.
This file follows the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format and semantic versioning.

 [v0.15.0] 
 - 2025-07-19 - Version Analysis
 
Adjustments made in model evaluation scripts ("evaluate_model.py" and "app.py") to allow to work with the new architecture of the model.

🟠 [Analysis] Overfitting detection in the latest training seasons
During the assessment of saved checkpoints (especially at more advanced times), progressive signs of overfitting were observed. The model began to identify regions outside the object of interest, including unwanted noises and leftovers in segmentation - behavior not observed during the initial training epochs.

This effect was progressive: the greater the number of times, the more the model "hallucinates" regions, extrapolating the real limits of the expected mask.

📌 Implications:

Clear signal that the model is memorizing patterns of the training set, including irrelevant visual artifacts or standards.

It can compromise the generalization of the model in real environments or unaware images.


- 2025-07-18
✅ Adjustments to the loss and TTA pipeline
Fixed the DiceLoss function

Added shape checking and compatibility between preds and targets to avoid broadcast errors.

Included safe conversion to float after binary comparison for class 1 masks.

Ensures that targets are correctly reduced if they have dimensions [B, 1, H, W].

Modified predict_with_tta to return logits

Created the predict_with_tta_logits function, which returns the mean of the logits before argmax, allowing direct use with loss functions such as CrossEntropyLoss and DiceLoss.

Maintained compatibility with the TTA structure, but now without compromising the backward compatibility of the training pipeline.

Standardization of outputs for use in loss analysis

Training adapted to always deliver logit tensors [B, C, H, W] for the loss criterion, regardless of the use of TTA.



[v0.14.0] - 2025-07-17 - Conditional Control via Settings
Improvements and Additions:

🔧 Added configurable flags at the beginning of the script (USE_TTA, USE_REFINEMENT, USE_FOCAL_LOSS, GAMMA_FOCAL) to enable/disable advanced behaviors in the pipeline in a simple and controlled manner.

✅ Automatic loss function selection based on the flag:

CrossEntropyLoss default.

FocalLoss with adjustable gamma (via GAMMA_FOCAL), activateable by flag.

🔁 Test-Time Augmentation (TTA) now optionally applicable in the inference phase:

Includes flips and rotations with automatic inversion.

Final prediction by averaging probability maps.

🧼 Mask smoothing via morphological closing/opening with cv2, also controllable by flag.

Applies morphological refinement to smooth contours and reduce jagged edges.

Motivation:
Allows modular experimentation, with a direct impact on validation metrics (IoU/Dice) without altering the core model or recoding sections. Flexibility is essential for controlled experimentation in the R&D cycle.


[v0.13.0] – 2025-07-16
⚙️ **Practical Training Adjustments**
- **Removed the use of `GradualWarmupScheduler`** due to an import error and dependency conflicts — will be re-evaluated in the future.
- **Kept the new `CosineAnnealingLR`** scheduler, promoting a smooth variation in the learning rate across epochs.

🧪 **Experimental Configuration**
- **Adjusted `early_stop_patience` to 40**, allowing the model greater exposure to the data before stopping training due to stagnation.

💡 Notes
- Testing with 20,000 512×512 images caused GPU memory overhead — revised strategy for smaller batches and progressive adjustments.


[v0.12.0]-2025-07-14

🚀 New Phase: Consistent Dataset and High-Resolution Input

🧼 Dataset Reconstruction
Dataset completely recreated from scratch, correcting critical flaws in image-to-mask matching.

Fixed a bug in the preprocessing script that could mix or swap masks between images.

Images and masks are now guaranteed to be aligned, with structural consistency and no label pollution.

📏 Resolution Increase
Input resolution increased from 256×256 → 512×512, allowing for better definition of shapes and contours.

Architecture adjusted to support the new dimensions while maintaining U-Net flow with skip connections.

🧠 Quality and Focus
Higher information density per image, promoting more refined learning.

Expected reduction of "coarse blocks" in predictions, with improvements in edges and spatial orientation.

💡 Notes
This new stage marks the transition from the exploratory phase to a more mature, validated pipeline aligned with best practices for deep segmentation.

Future validations will include overlay visualizations, class-based metrics, and qualitative comparisons between versions.


🎯 Segmentation and robustness refinements

🧼 Correction of masks with invalid values
Detected critical error: Some masks had values 2, causing failures in metrics (Valueerror: Unseen Labels).

Applied solution:

Converted masks for numpy, binarized with mask = (mask> 127) .astype (np.uint8), ensuring only values 0 and 1.

Posterior conversion with Torch.from_numpy (...). Long () for use in the model.

Validation of integrity with NP.unique () and explicit error if unexpected values are found.

🧪 Pipeline diagnosis
Added temporary inspection of unique values to masks during __getitem __ () to identify out of standard data.

📉 Correction of metrics
Sklearn's Jaccard_Score use adjustment for multiclasses scenarios:

Replaced averag = 'binary' by averag = 'macro' to avoid error valueerror: target is multiclass ....

🗃️ Dataset and loader
Transformations maintained with .convert ("L") to ensure gray scale images and masks despite posterior binarization.

Confirmation of data alignment between image and mask by direct checking in the loader.

🛠️ DataSet Stability
Implemented explicit check of the mask on the disc, avoiding silent failures.

Pipeline preparation to detect problems early during data loading.

💡 Observations
Despite refinements, the model still has rude segmentations ("blocks"), but already indicating directionality consistent with the object (car).

Next steps include augmentation adjustments, expansion of the receptive Field and improvement in contour capacity.


[v0.11.0] – 2025-07-13
🔧 Pipeline and Logging Refactoring
📄 Created the save_report() method to save training logs to a file continuously and securely (replaces print for traceability in production).

🧠 Moved the num_epochs, checkpoint_interval, and early_stop_patience parameters to config.py to centralize experiment configuration.

✅ Removed the fixed use of print() during training, facilitating use on clusters, remote notebooks, and reproducibility.

🗂️ Updated train.py to run with zero dependency on manual changes: everything is configurable via config.

🧮 Metric Adjustments
✅ Fixed and stabilized IoU and Dice Score calculations after each epoch, with automatic logging in the final report.

🧪 Dice_score calculated via np.logical_and and np.logical_or for greater accuracy and consistency with academic metrics.

📊 Automatic saving of metrics graphs:

training_loss.png

training_val_accuracy.png

iou_history.png

dice_history.png

🧬 Dataset
🔁 Maintained robust preprocessing and mask binarization.

💡 Validation of continuous image-mask alignment externally (via separate script) before training.

🚀 Certification Preparation
🎯 Stable modular structure, suitable for submission to platforms such as Hugging Face.

🔒 Centralized logging ensures hardware traceability (CUDA and Torch versions recorded at the start of training).


[v0.10.0] – 2025-07-12
🧾 Training Pipeline Refinements and Automated Reporting

📋 Modularization and Centralized Configuration
🔧 Critical parameters now defined via scripts/config.py:

num_epochs = 250

checkpoint_interval = 15

early_stop_patience = 60

Easy adjustments without changing the main training code

📝 Execution Log with save_report()
✅ Added save_report(row) method to log:

Library versions (Torch, CUDA)

Progress per epoch (Loss, Accuracy, IoU, Dice)

Training Start and End

Final Performance Summary

📁 Logs automatically saved in config.report_file, allowing historical tracking and auditing of the execution

💡 Notes
Code now ready for automated cluster executions, CI/CD, or continuous validation pipelines.

Standardization facilitates future integration with TensorBoard, Gradio, or custom dashboards.


[v0.9.0] – 2025-07-10
🎯 Certification and Standards Compliance

📦 Class Structure and Compliance
🔄 Inverted dataset classes to follow the conventional pattern:

0: Background

1: Object

Avoids confusion in standard metrics such as CrossEntropy and Jaccard

🧠 ReLU in Skip Connections
🚀 Added F.relu(...) activations after up + skip sums in U-Net, improving the ability to learn nonlinearities between blocks

Fixes linear behavior of activations in the decoding phase

⚖️ Adjusted Class Balance
⚙️ compute_class_weights() now uses a more robust normalization formula:

weights = class_counts.sum() / (2.0 * class_counts + 1e-6)
weights = weights / weights.sum()
Avoids overfitting of the minority class without distorting learning.

📊 Advanced Training Metrics
✅ Calculation of IoU (Jaccard) and Dice Score on the validation set per epoch:

sklearn's jaccard_score()

Dice with intersection / union using NumPy

📉 Stored as iou_history and dice_history, with graphs saved via matplotlib

🖼️ Metrics Visualization
New graphs:

iou_history.png

dice_history.png

All graphs are saved directly, maintaining compatibility with CUDA/headless training environments.

🧪 Stability and Diagnostics
Fixed bug TypeError: Cannot interpret '-1' as a data type caused by incorrect types in np.concatenate of PyTorch arrays with .astype('int')

Now guaranteed Validation data should be np.uint8 to avoid conflicts.

💡 Notes
Model now follows the Hugging Face Vision Certification metrics standard.

Validated pipeline with clear training/validation separation, reliable metrics, and an extensible multiclass structure.


[v0.8.0] – 2025-07-09
🧠 Pipeline Architecture and Reconstruction
✅ Replaced the binary output model with a multiclass architecture (num_classes=2) with CrossEntropyLoss and softmax, allowing future expansion to multi-class segmentations.

🧪 New composite loss function:

Implemented custom DiceLoss with smooth=1e-6 for greater sensitivity to contours and thin areas

Combined with class-weighted CrossEntropyLoss: loss = 0.5 * CrossEntropy + 0.5 * DiceLoss

📊 Dynamic calculation of class weights:

Added compute_class_weights() method to balance the loss based on the actual pixel frequency per class in the dataset

Replaces previous fixed weights, automatically adapting to new datasets

🧬 Dataset and Preprocessing
🖼️ Updated Dataset SegmentationDataset:

Robust loading with mask presence check

Image and mask conversion to grayscale

Binarized masks with threshold (mask > 127).long() to ensure values ​​{0, 1}

🎨 Transformations:

Applied Resize (256×256) and normalized with mean=[0.5], std=[0.5] for single-channel input

🏗️ ResNetUNet Model
🔁 Reconstructed architecture based on resnet50 (pretrained=True):

Adapted conv1 for single-channel input

Skip connections with residual sum between encoder and decoder

Final upsample with nn.Upsample(scale_factor=2) to restore original resolution

🏋️ Training and Monitoring
📈 Training with:

AdamW with lr=1e-4 and weight_decay=1e-4

StepLR scheduler with gamma=0.5 every 10 epochs

Early stopping with patience=60

Checkpoints saved every 15 epochs

Automatic saving of the best model based on train_accuracy

📊 Metrics:

Pixel accuracy for training and validation

History of loss and accuracy by epoch

Graphs saved as .png with plt.savefig() (without plt.show())

💡 Notes
Model now prepared for multiclass segmentations with greater stability

More robust and modular pipeline, with a clear separation between architecture, dataset, loss, and training

Structure ready for integration with metrics such as IoU, F1-score, and visualization with TensorBoard


[v0.7.0] – 2025-07-07
🧪 Advanced Binary Segmentation
✅ Modified architecture: ResNetUNet model adjusted for single-channel output (num_classes=1), with sigmoid applied in the final step — prepared for smooth binary segmentation.

🧠 Masks reformatted in the dataset: converted to float32 with shape [1, H, W] and binarized via threshold, optimizing compatibility with BCEWithLogitsLoss.

🎯 New Composite Loss Function
➕ Implemented custom Dice Loss to improve learning of contours and thin areas, combined with BCEWithLogitsLoss in equal weight.

🧬 Formula applied: 0.5 * BCE + 0.5 * Dice, increasing the model's sensitivity to the real geometry of the segmented objects.

🧮 Improved Pixel-Wise Evaluation
📏 Pixel accuracy adjusted to consider sigmoid and binary threshold (0.5) in predictions before comparing with masks — makes the calculation more faithful to the purpose of segmentation.

💡 Observations
Model now captures smoother contours, reducing "square" behavior.

Code now ready for integration with advanced metrics such as IoU, Precision/Recall per class, and image visualization with matplotlib or TensorBoard.


[v0.6.0] – 2025-07-07
🧠 Training Pipeline Refinement
🔁 Training now separated by training and validation: automatic splitting of the SegmentationDataset into 80/20 to monitor generalization.

📊 Validation implemented per epoch with accuracy calculation on the validation set; metric used for early stopping and best_model.pt selection.

📈 Generalization and Robustness
🌈 Added augmentation transformations via RandomHorizontalFlip and RandomRotation on the training set, to make the model more resistant to visual variations.

⏳ Early stopping increased: early_stop_patience increased from 20 to 60 epochs, giving more room for progressive learning.

🔁 Hyperparameters and Regularization
📉 Added weight_decay=1e-5 in the Adam optimizer for lightweight L2 regularization.

🎯 Best model metric changed: now best_model.pt saves based on the best validation accuracy, not just training.

📊 Results visualization
🖼️ New graph generated training_val_accuracy.png showing the evolution of validation accuracy over epochs.

📊 Graphs saved with plt.savefig() after try/except, avoiding failures in environments with graphics rendering issues via CUDA.

💡 Observations
Model showed qualitative improvement in segmentation with smoother and more responsive contours — previous squares started to follow the car's rotation, indicating spatial learning.

Structure ready for future calculation of IoU per class and integration with TensorBoard, if necessary.


## [v0.5.0] – 2025-07-05
### ⚒️ Critical Data Alignment Fixes
Fixed mask file mismatch: Masks were stored as .png, but dataset loader expected .jpg extension — caused incorrect or failed loading

### 🧠 Applied patch via .replace('.jpg', '.png') in dataset loader to ensure proper image-mask pairing

Added FileNotFoundError checks during __getitem__ to avoid silent failures and improve debugging clarity

### 🧠 Dataset Refinements
Ensured matching Resize(256×256) transformations for both image and mask, using transforms.functional for consistency

Binarization of masks confirmed to produce only {0,1} values, avoiding grayscale range leakage

Validated with np.unique() on mask tensors — clean value range critical for CrossEntropyLoss


---

## [v0.4.0] – 2025-07-04

### ✅ Major Improvements

- Added **checkpoint saving** every `N` epochs during training, configurable via a new `checkpoint_interval` parameter.
- Implemented **Early Stopping** based on pixel accuracy, with a configurable patience (`early_stop_patience`) to avoid overfitting.
- Final model now saved **twice**:  
  - `best_model.pt`: Only weights, for inference/embedded use  
  - Full model (`torch.save(model, ...)`) at the end, for future reloading

### ⚠️ Critical Bug Diagnosed and Resolved

- **Symptom:** Model was training with no improvement; accuracy stuck; no learning observed
- **Diagnosis:** Masks loaded from `.jpg` were using full grayscale range `[0, 1, ..., 255]` instead of binary values `[0, 1]`
- **Fix:** Added diagnostic checks using `np.unique` to validate mask classes; incorporated a preprocessing step to binarize masks

### 🧪 Experimental Enhancements

- Updated model evaluation interface (`evaluate_model.py`) for batch testing via folder traversal
- Separated Gradio demo (`app.py`) for certification usability evaluation
- Integrated plotting of loss and accuracy with graceful error handling (wrapped in `try/except`)

### 🧠 Observations

- Problem with **matplotlib crashing** due to CUDA context when using `plt.show()` after training; workaround applied with `plt.savefig()` only
- CUDA kernel mismatch on certain environments using **dual RTX 4060** detected as 3060 — resolved by adjusting `torch` + `nvidia-driver` stack (manual)
- Added check to confirm training device (`config.device`) and `torch.cuda.get_arch_list()` for future reproducibility

### 🤝 Acknowledgements

- Much of the model debugging was assisted by real-time reasoning and exploration with **ChatGPT**, especially around mask encoding and loss mismatch.
- Initial development relied on **GitHub Copilot**, with ChatGPT joining later to refator, modularize, and refine robustness for submission.

---

## [v0.3.0] – 2025-06-22

🐛 Bug Fixes  
- Fixed `RuntimeError: only batches of spatial targets supported (3D tensors)` caused by mask dimensions  
- Applied `.squeeze(1)` to target tensors before passing to `CrossEntropyLoss`, ensuring correct shape `(B, H, W)`  
- Root cause: mask loaded with shape `(B, 1, H, W)` instead of `(B, H, W)`

👁️ Observations  
- Issue identified during initial model training with grayscale images and ResNet-based U-Net  
- Fix reduces debugging time from hours to seconds — thanks to a productive collaboration with Microsoft Copilot 🧠

---

## [v0.2.0] – 2025-06-22

### 🔧 Project Restructure
- Fully reorganized project files to reflect a modular and scalable architecture
- Added new root folders:  
  - `DataSet/Cow_Segmentation_Dataset/` to centralize all data and annotations  
  - `scripts/Dataset/` for preprocessing and data preparation logic  
  - `scripts/Segmentation/` for training, evaluation, and model utilities  
  - `scripts/Segmentation/Future/` to house experimental/embedded extensions

### 📑 Documentation Updates
- Updated `README.md` to match new folder organization and include a **Future Work** section
- Updated `model_card.md` to reflect modular design and embedded plans

### 💡 Future-Ready Additions
- Introduced experimental script `train_embedded_explicit_model.py` for ONNX export and embedded deployment (not yet validated)

---

## [v0.1.0] – Initial Release

### 🚀 Baseline Functionality
- Preprocessing scripts for grayscale mask generation and dataset formatting
- Training and evaluation scripts for custom segmentation model
- Initial model card and license

---

## [v0.0.1] – Project Start

- Initial discussion on using ResNet as an encoder in U-Net
- Creation of an example synthetic dataset
- Structuring of the basic inference script
- Validation of the visual pipeline and preprocessing strategy
