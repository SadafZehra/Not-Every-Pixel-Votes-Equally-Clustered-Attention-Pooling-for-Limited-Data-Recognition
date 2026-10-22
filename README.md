# Not Every Pixel Votes Equally: Clustered Attention Pooling for Limited-Data Recognition

At the end of most CNNs, **global average pooling (GAP)** gives every spatial position the same vote. Background pixels count as much as the object. This project asks whether **letting the network choose where to look** (attention pooling) or **grouping positions into regions first** (clustered attention pooling) helps, especially when training data is scarce.

All three heads sit on the same ResNet backbone, built from scratch and trained on CIFAR-100 with 10% and 100% of the training data.

## Key results

CIFAR-100 test set (10,000 images), one seed.

| Head | Data | Top-1 ↑ | Top-5 ↑ | NLL ↓ | ECE (%) ↓ | Occluded top-1 ↑ | Head params |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GAP (baseline) | 10% | 39.16 | 64.07 | 2.71 | **3.45** | 29.25 | 25.7K |
| Attention pooling | 10% | **40.47** | 65.47 | 2.66 | 7.67 | 29.50 | 306.0K |
| Clustered attention pooling | 10% | 40.13 | **65.48** | 2.66 | 7.10 | **29.62** | 306.5K |
| GAP (baseline) | 100% | **74.45** | **92.79** | **1.01** | 5.88 | 53.94 | 25.7K |
| Attention pooling | 100% | 74.11 | 92.55 | 1.02 | 6.01 | 53.91 | 306.0K |
| Clustered attention pooling | 100% | 73.87 | 92.18 | 1.03 | **5.75** | **54.08** | 306.5K |

*10% = 50 images per class, 100% = 500 images per class. ECE = expected calibration error. Occluded = accuracy when a 12×12 patch of the image is blanked out.*

## Findings

1. **Attention helps a little when data is scarce.** With 50 images per class, attention pooling gains **+1.31 points** top-1 over GAP (clustered: +0.97).
2. **But it makes the model overconfident.** Calibration error more than doubles, from **3.45% to 7.67%**. The model gets slightly more answers right but becomes much less honest about how sure it is.
3. **With full data the advantage disappears.** All three heads land at about 74% top-1; differences under 0.5 points are within normal seed-to-seed variation on CIFAR-100.
4. **Clustering adds nothing on top of plain attention**, while costing more training time (12.4 vs 10.6 minutes on the full data).
5. **Grad-CAM maps look almost the same for all three heads** (about 48% of the evidence falls in the top 20% of pixels for each), so the heads change how features are pooled more than where the backbone looks.

The takeaway: a learned pooling head is not a free win. Its small accuracy gain in the low-data regime has to be weighed against worse calibration.

## Method

**Backbone.** A CIFAR-style ResNet built from scratch: a 3×3 stem and three stages of two basic blocks with 64, 128 and 256 channels (2.78M parameters). Only stages two and three downsample, so the final feature map is 8×8, giving 64 positions for the pooling heads.

**Pooling heads.** Each head maps the `[B, C, 8, 8]` feature map to class logits.
- *GAP:* the plain mean over all 64 positions.
- *Attention pooling:* learned queries (4 heads) attend over the 64 positions; the plain mean is kept as a residual, so the head can fall back to GAP.
- *Clustered attention pooling:*
  1. adds a learned position embedding and LayerNorm to the 64 position tokens;
  2. picks 4 starting centroids by farthest-point selection and runs 3 steps of differentiable soft k-means;
  3. lets one learned query attend over the 4 centroids and the 64 positions (a type embedding marks which is which).
  Clustering runs in float32 even under mixed precision.

**Training recipe.** SGD with Nesterov momentum 0.9, learning rate 0.1, linear warm-up (2 epochs) then cosine decay, weight decay 5e-4 on weights only (not on norms, biases, embeddings or queries), label smoothing 0.1, batch 128, mixed precision. Standard CIFAR augmentation (random crop with padding, horizontal flip). 100 epochs on the 10% split, 30 epochs on the full data.

**Analysis.** Top-1/top-5 accuracy, NLL, expected calibration error, robustness to a 12×12 occlusion, and **Grad-CAM** (Selvaraju et al., 2017) maps, plus visualizations of where the attention query looks and how the clustered head splits the image into 4 regions without any part labels.

## How to run

1. Open the notebook in Google Colab with a GPU runtime and run all cells. CIFAR-100 is downloaded through torchvision.
2. Set `FAST_DEV_RUN = True` for a quick check (1 epoch on the 10% split, 1,000 test images).
3. Outputs are written to `outputs_part3/`. A full run takes roughly 45 minutes on a T4.

The notebook also includes unit tests for every head before training starts.

## Main libraries

PyTorch · torchvision · NumPy · pandas · Matplotlib

## Limitations

One seed per configuration. A proper conclusion on the +1.3 point gain needs several seeds; temperature scaling could also be tested to see whether the calibration gap can be closed after training.

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
