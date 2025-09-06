# ODA-GAN: Orthogonal Decoupling Alignment GAN Assisted by  Weakly-supervised Learning for Virtual Immunohistochemistry Staining
This repository contains selected core implementations accompanying our paper on HE-to-IHC stain transfer.  
At this stage, we are releasing only the essential components to illustrate the methodology.  
The full training and inference codebase is being organized and will be released in a future update.
**Contact:** For any questions or clarifications, please contact: [wangtong8868@gmail.com](mailto:wangtong8868@gmail.com)
## Methods Overview
We highlight two main components of our framework:
- **Idea 1**: Multi-layer Domain Alignment (MDA)
- **Idea 2**: Dual-stream PatchNCE (DPNCE)

Below we provide key code snippets to illustrate their implementation.
## Ideas 1: Multi-layer Domain Alignment
We propose an MDA module to minimize the distribution discrepancy between virtual and real image features to enable more realistic staining.
### main code
```py
def gaussian_kernel(x, y, sigma=1.0):
    """RBF Kernel function."""
    dist = torch.cdist(x, y, p=2)
    return torch.exp(-dist.pow(2) / (2 * sigma ** 2))

def compute_mmd_loss(batch_features_x, batch_features_y, sigma=1.0):
    """
    Compute MMD loss between two batches of features.

    Args:
    - batch_features_x: tensor of shape [n, d], representing a batch of features.
    - batch_features_y: tensor of shape [m, d], representing another batch of features.
    - sigma: float, bandwidth for RBF kernel.

    Returns:
    - mmd_loss: MMD loss scalar.
    """
    # Compute the kernel values
    k_x_x = gaussian_kernel(batch_features_x, batch_features_x, sigma)
    k_y_y = gaussian_kernel(batch_features_y, batch_features_y, sigma)
    k_x_y = gaussian_kernel(batch_features_x, batch_features_y, sigma)
    
    # Compute MMD loss
    mmd_loss = k_x_x.mean() + k_y_y.mean() - 2 * k_x_y.mean()
    
    return mmd_loss


from collections import deque
"""
Maintain dynamic feature queues for domain adaptation loss.

Args:
- queue_size: int, maximum length of each queue (oldest features will be dropped when exceeded).
- layers: str, indices of feature extraction layers from the generator (e.g., '0,4,8,12,16').
- num_layers: int, number of layers to align between domains.
- num_patches: int, number of patches sampled from each feature map.
- masks: list of tensors, each of shape [num_patches], binary mask where 1 indicates tumor and 0 indicates normal.
- feat_fake: list of tensors, fake domain features extracted at selected layers.
- feat_real: list of tensors, real domain features extracted at selected layers.

Returns:
- fake_tum_queues, fake_nor_queues, real_tum_queues, real_nor_queues:
  lists of length `num_layers`, where each element is a deque storing historical
  feature batches for the corresponding category (tumor or normal) and domain (fake or real).
"""
queue_size = 1024
layers = '0,4,8,12,16'
num_layers = 5
num_patches = 256

fake_tum_queues = [deque(maxlen=queue_size) for _ in range(num_layers)]
fake_nor_queues = [deque(maxlen=queue_size) for _ in range(num_layers)]
real_tum_queues = [deque(maxlen=queue_size) for _ in range(num_layers)]
real_nor_queues = [deque(maxlen=queue_size) for _ in range(num_layers)]

feat_fake = netG(fake, layers, encode_only=True)
feat_real = netG(real, layers, encode_only=True)

feat_fake_pool, sample_ids = netF(feat_fake, num_patches, None)
feat_real_pool, _ = netF(feat_real, num_patches, sample_ids)

for i in range(num_layers):
    ids = sample_ids[i]
    tum_mask = masks[i][ids] == 1
    nor_mask = masks[i][ids] == 0
    fake_tum_queues[i].append(feat_fake_pool[i][tum_mask])
    fake_nor_queues[i].append(feat_fake_pool[i][nor_mask])
    real_tum_queues[i].append(feat_real_pool[i][tum_mask])
    real_tum_queues[i].append(feat_real_pool[i][nor_mask])

def calculate_DA_loss(fake_tum_queues, fake_nor_queues, real_tum_queues, real_tum_queues):
    """
    Compute domain adaptation loss with dynamic queues.

    Args:
    - fake_tum_queues: list of deques, storing fake tumor features per layer.
    - fake_nor_queues: list of deques, storing fake normal features per layer.
    - real_tum_queues: list of deques, storing real tumor features per layer.
    - real_nor_queues: list of deques, storing real normal features per layer.

    Returns:
    - da_loss: scalar, averaged domain adaptation loss across layers.
    """
    total_loss = 0
    for i in range(num_layers):
        loss_tum = compute_mmd_loss(fake_tum_queues[i], real_tum_queues[i])
        loss_nor = compute_mmd_loss(fake_nor_queues[i], real_tum_queues[i])
        total_loss += 0.5 * (loss_tum + loss_nor)
    return total_loss / num_layers

```

## Ideas 2: Dual-stream PatchNCE
We propose a DPNCE loss to resolve contrastive learning contradictions by decoupling image features into staining-related and unrelated components, processing them separately to ensure pathological consistency.
### main code
```py
import mctorch.nn as mnn

class ManifoldMLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super(ManifoldMLP, self).__init__()
        self.conv1 = mnn.rConv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1, weight_manifold=mnn.Stiefel)
        self.relu = nn.ReLU()
        self.conv2 = mnn.rConv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1, weight_manifold=mnn.Stiefel)
    
    def forward(self, x):
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x.squeeze(2).squeeze(2)

class DWFC(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(DWFC, self).__init__()
        # Depth-wise Encoder with 3 convolutional blocks
        self.depthwise_encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, num_classes)
        self.fc2 = nn.Linear(in_channels, num_classes)
        # Learnable mask m with sigmoid to constrain between 0 and 1
        self.mask = nn.Parameter(torch.randn(1, in_channels, 1, 1))  # Broadcastable mask
    def forward(self, x):
        x = self.depthwise_encoder(x)
        m = torch.sigmoid(self.mask)  # Constrain mask to be between 0 and 1
        # Split features using m and 1 - m
        features_m = x * m 
        features_1m = x * (1 - m)

        pooled_m = self.gap(features_m).view(features_m.size(0), -1)  # (batch_size, in_channels)
        pooled_1m = self.gap(features_1m).view(features_1m.size(0), -1)

        out_m = self.fc1(pooled_m)  # Prediction from selected features
        out_1m = self.fc2(pooled_1m)  # Prediction from remaining features

        return features_m, features_1m, out_m, out_1m

# Define a ModuleList to process a list of 5 feature maps with 5 DWFC
class DWFCModuleList(nn.Module):
    def __init__(self, in_channels_list, gpu_ids):
        super(DWFCModuleList, self).__init__()
        self.device = torch.device(f'cuda:{gpu_ids[0]}')
        self.dwfc_list = nn.ModuleList([DWFC(in_channels).to(self.device) for in_channels in in_channels_list])

    def forward(self, feature_maps):
        outputs = []
        for i, feature_map in enumerate(feature_maps):
            output = self.dwfc_list[i](feature_map)
            outputs.append(output)
        return outputs

def define_netC(in_channels_list, gpu_ids):
    model = DWFCModuleList(in_channels_list, gpu_ids)
    return model

# Training setup
self.netC = networks.define_C([f.shape[1] for f in feat], self.gpu_ids)
self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
self.optimizers.append(self.optimizer_C)

def compute_CE_loss(self, labels):
    """
    Compute DPNCE loss:
    - For each layer, compute CE on staining-related stream (loss_m) and unrelated stream (loss_1m).
    - Contrast them by subtraction (loss_m - loss_1m).
    - Average across layers and domains (real_A, real_B).
    """
    n_layers = len(self.nce_layers)
    feats_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
    feats_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
    outs_real_A = self.netC(feats_real_A)
    outs_real_B = self.netC(feats_real_B)

    # netC out: features_m, features_1m, out_m, out_1m
    pred_A_m = [p[2] for p in outs_real_A]
    pred_A_m = torch.cat(pred_A_m, dim=0)
    pred_A_1m = [p[3] for p in outs_real_A]
    pred_A_1m = torch.cat(pred_A_1m, dim=0)
    loss_A_m = self.criterionC(pred_A_m, labels)
    loss_A_1m = self.criterionC(pred_A_1m, labels)
    loss_A_CE = (loss_A_m - loss_A_1m) / n_layers

    pred_B_m = [p[2] for p in outs_real_B]
    pred_B_m = torch.cat(pred_B_m, dim=0)
    pred_B_1m = [p[3] for p in outs_real_B]
    pred_B_1m = torch.cat(pred_B_1m, dim=0)
    loss_B_m = self.criterionC(pred_B_m, labels)
    loss_B_1m = self.criterionC(pred_B_1m, labels)
    loss_B_CE = (loss_B_m - loss_B_1m) / n_layers

    loss_CE = 0.5 * (loss_A_CE + loss_B_CE)

    return loss_CE
```
