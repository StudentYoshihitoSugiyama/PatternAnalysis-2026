import torch
import torch.nn as nn

# ==========================================
# GFNet Hyperparameters (Global Configuration)
# ==========================================
IMG_SIZE = 256
IN_CHANS = 1          # Greyscale images
PATCH_SIZE = 16       # Spatial patch size: 256 / 16 = 16 patches per axis
EMBED_DIM = 256       # Hidden embedding dimension
DEPTH = 6             # Number of GFNet Transformer blocks
MLP_RATIO = 4.0       # MLP expansion ratio
NUM_CLASSES = 2       # NC (0) vs AD (1)
DROPOUT_RATE = 0.1    # Dropout probability


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.module):
    """
    ResNet-18 Model tailored for ADNI dataset (256x256 images).
    
    Neuron & Architectural Structure Suggestion:
    - Initial Layer: 3x3 Convolution (64 feature channels) with stride 1 to retain 32x32 spatial resolution.
    - Residual Layers (Layer 1 to 4): Feature maps expand from 64 -> 128 -> 256 -> 512 channels.
    - Pre-logits Pooling: Adaptive Global Average Pooling collapses 512 feature channels down to a 512-dimensional vector.
    - Regularization: Dropout layer (p=0.3) applied before classifier to prevent co-adaptation.
    - Optimal Classifier Neurons:
      - 512 input neurons (matching the 512 residual output feature channels).
      - 2 output neurons (mapping to the 2 class logits).
    """
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        # 512 input neurons -> 2 output class neurons
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class PatchEmbed(nn.Module):
    """
    Splits 2D image into patches and linearly projects them into embedding dimension.
    """
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, embed_dim=EMBED_DIM):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, H_grid, W_grid) -> (B, H_grid, W_grid, D)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class GlobalFilter(nn.Module):
    """
    Global Filter layer performing 2D FFT, element-wise multiplication with learnable
    complex frequency weights, and 2D inverse FFT.
    """
    def __init__(self, dim=EMBED_DIM, h=IMG_SIZE // PATCH_SIZE, w=IMG_SIZE // PATCH_SIZE):
        super(GlobalFilter, self).__init__()
        self.h = h
        self.w = w
        # Real FFT along width dimension produces w // 2 + 1 non-redundant frequencies
        self.complex_weight = nn.Parameter(
            torch.randn(h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, H, W, C = x.shape
        # Convert to float32 for FFT calculation
        x = x.to(torch.float32)

        # 2D Real FFT along spatial axes (H, W)
        x_fft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Form complex weights from the 2 real channels
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight

        # Inverse 2D Real FFT back to spatial domain
        x = torch.fft.irfft2(x_fft, s=(H, W), dim=(1, 2), norm='ortho')
        return x


class Mlp(nn.Module):
    """
    Standard Feed-Forward Network (MLP) used in Transformer blocks.
    """
    def __init__(self, in_features=EMBED_DIM, hidden_features=None, out_features=None, drop=DROPOUT_RATE):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * MLP_RATIO)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GFNetBlock(nn.Module):
    """
    GFNet Transformer Block: LayerNorm -> GlobalFilter -> LayerNorm -> Mlp with residual connections.
    """
    def __init__(self, dim=EMBED_DIM, h=IMG_SIZE // PATCH_SIZE, w=IMG_SIZE // PATCH_SIZE,
                 mlp_ratio=MLP_RATIO, drop=DROPOUT_RATE):
        super(GFNetBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim=dim, h=h, w=w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # Global filter sub-layer with residual connection
        x = x + self.filter(self.norm1(x))
        # MLP sub-layer with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class GFNet(nn.Module):
    """
    Global Filter Network (GFNet) tailored for 256x256 greyscale ADNI images.
    """
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
                 num_classes=NUM_CLASSES, embed_dim=EMBED_DIM, depth=DEPTH,
                 mlp_ratio=MLP_RATIO, drop_rate=DROPOUT_RATE):
        super(GFNet, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        h, w = self.patch_embed.grid_size

        # Learnable 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, h, w, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks with global Fourier filters
        self.blocks = nn.ModuleList([
            GFNetBlock(dim=embed_dim, h=h, w=w, mlp_ratio=mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding: (B, C, 256, 256) -> (B, H_grid, W_grid, D)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Global filter blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Global average pooling over spatial grid dimensions (H_grid, W_grid)
        x = x.mean(dim=(1, 2))
        logits = self.head(x)
        return logits

