# Explanations
## Text-Driven Image Segmentation with SAM 2

### Pipeline
1. Load image.
2. Take a text prompt from user.
3. Use **GroundingDINO** to convert text into bounding box seeds.
4. Feed seeds into **SAM 2 (Segment Anything Model 2)**.
5. Display the segmented mask overlay on the image.

## Vision Transformer (ViT) on CIFAR-10

The notebook includes training, evaluation, and automated experiments.

### Model Configuration (Best Model)
- **Image size:** 32×32
- **Patch size:** 4×4
- **Embedding dimension:** 256/512
- **Number of layers:** 6
- **Number of attention heads:** 8
- **MLP dimension:** 512
- **Dropout:** 0.1
- **Optimizer:** AdamW with learning rate 3e-4
- **Batch size:** 128
- **Epochs:** 20

### Test Accuracy Results

| Experiment | Test Accuracy |
|------------|---------------|
| Baseline (4×4, 6 layers, 256 embed) | 73.38% |
| Patch size 8×8 | 65.40% |
| Shallow (layers=4) | 73.06% |
| Wider embedding (512) | 73.40% |
| No data augmentation | 63.94% |
| Baseline, non-overlapping, AdamW | 66.10% |
| Overlapping patches, AdamW | 72.54% |
| Non-overlapping, Adam | 65.61% |

### Analysis and Observations

#### 1. Patch Size
- Small patches (4×4) performed best (73.4%).
- Larger patches (8×8) significantly reduced accuracy (65.4%) — fine details are lost on low-resolution images.

#### 2. Depth vs. Width
- Reducing layers from 6 → 4 gave only a small drop (73.3% → 73.1%), suggesting depth was not the main limiting factor.
- Increasing embedding dimension to 512 did not improve performance beyond baseline (73.4%).

#### 3. Data Augmentation
- Removing augmentation dropped accuracy sharply (73.4% → 63.9%), showing augmentation is critical for generalization.

#### 4. Patch Overlap
- Overlapping patches with AdamW improved performance over the non-overlapping AdamW baseline (72.5% vs 66.1%).
- Suggests overlap can help capture spatial continuity when other factors are suboptimal.

#### 5. Optimizer
- AdamW clearly outperformed Adam (66.1% vs 65.6% for non-overlapping).
- This aligns with expectations from transformer literature where weight decay regularization is crucial.

#### 6. Key Takeaways
- The current setup underperforms state-of-the-art ViTs on CIFAR-10 (expected ~85–90%+).
- Performance gaps are likely due to:
  - Shorter training schedules (20 epochs vs. 100+ for SOTA)
  - Weaker augmentation strategies (needs RandAugment, CutMix, Mixup)
  - Lack of pretrained weights or self-supervised pretraining
- Nevertheless, the framework works correctly and clearly demonstrates how each design choice impacts model performance.
- Small patches and data augmentation are the most critical factors for this low-resolution dataset.