import torch
import torch.nn as nn


###---------- Helper blocks ----------###
# For intermediate CNN
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_hardswish=False):
        super().__init__()
        activation = nn.Hardswish if use_hardswish else nn.ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.activation = activation(inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x) + self.shortcut(x))


# For advanced CNN
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, use_se=False):
        super().__init__()
        self.use_se = use_se
        mid_channels = in_channels
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(inplace=True)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )
        
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_se:
            x = self.se(x)
        return x


# For MobileNetV3-like CNN
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class ConvBNActivation(nn.Module):
    """Standard convolution block with BatchNormalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, activation=nn.Hardswish):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual Block with Squeeze-and-Excitation"""
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, stride=1, se_ratio=0.25, activation=nn.Hardswish):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Calculate expansion channels
        expanded_channels = _make_divisible(in_channels * expand_ratio, 8)
        
        # Build block
        layers = []
        
        # Expansion phase (pointwise conv)
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_channels, expanded_channels, kernel_size=1, activation=activation))
        
        # Depthwise phase
        layers.append(ConvBNActivation(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                                      stride=stride, groups=expanded_channels, activation=activation))
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            layers.append(SEBlock(expanded_channels, reduction_ratio=expanded_channels // se_channels))
        
        # Projection phase (pointwise conv)
        layers.append(ConvBNActivation(expanded_channels, out_channels, kernel_size=1, activation=None))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function ensures that all layers have a channel number that is divisible by divisor.
    It is taken from the original tensorflow repository:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Ensure small round down
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
###-----------------------------------###


###------- Model Architectures -------###
class BasicCNN(nn.Module):
    '''
    MOTIVATION:
    Convolutional layers with ReLU activations are the minimum 'standard' for CNNs.
    Batch normalization layers have become part of this 'standard' for more stable training.
    For downsampling strided convolutions have become more common, especially in earlier layers.
    Using AdaptiveAvgPool prevents hardcoding of dimensional transformations, which depend on input size otherwise.
    '''
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Downsample through strided convolutions
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample through pooling
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class IntermediateCNN(nn.Module):
    '''
    MOTIVATION:
    Use of Hardswish activations instead of ReLU, close in computational costs while with most 
    benefits of the Swish activation over ReLU (smoother so more easy to optimize, reduced risk of dead neurons)
    Addition of residual connections to reduce vanishing gradients, allowing for features to remain
    relevant in deeper layers of the model.
    These additions are interesting as they could improve convergence speed and/or final accuracy
    without adding to much computational complexity and model size.
    '''
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Lightweight residual block
            ResidualBlock(64, 128, use_hardswish=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdvancedCNN(nn.Module):
    '''
    MOTIVATION:
    Use of depthwise separable convolutions, inspired by the MobileNet family. Drastically reduces parameter
    count and computation by splitting spatial filtering and channel mixing.
    Introduction of Squeeze-and-Excitation blocks to 'recalibrate' channel-wise feature responses, boosting the
    relevance of 'important' features while suppressing irrelevant ones.
    Test whether these efficiency-accuracy trade-offs are worthwile in the given setting.
    '''
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNActivation(3, 32, kernel_size=3, stride=2, activation=nn.Hardswish),
            
            DepthwiseSeparableConv(32, 64, activation=nn.Hardswish),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DepthwiseSeparableConv(64, 128, activation=nn.Hardswish, use_se=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DepthwiseSeparableConv(128, 256, activation=nn.Hardswish, use_se=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MobileNetV3(nn.Module):
    '''
    MOTIVATION:
    Inverted residual blocks encourage the learning of 'richer' representations with minimal cost.
    The use of expansion ratios allows for bottlenecks in the architecture, encouraging more efficient
    and compact feature learning, leading to improved generalization and usability.
    Balance the use of ReLU and Hardswish for best fit to given setting.
    SE block tuning to prevent unnecessary computational overhead if not necessary for some layers.
    Channels are made divisible by 8 for hardware efficiency.
    '''
    def __init__(self, num_classes=1000, width_mult=1.0, mode='huge'):
        super().__init__()
        
        # Define configuration based on mode (small, large, or huge)
        if mode == 'small':
            # Configuration for MobileNetV3-Small
            inverted_residual_setting = [
                # k, exp, c, se, act, s
                [3, 1, 16, True, 'RE', 2],     # 0
                [3, 4.5, 24, False, 'RE', 2],  # 1
                [3, 3.67, 24, False, 'RE', 1], # 2
                [5, 4, 40, True, 'HS', 2],     # 3
                [5, 6, 40, True, 'HS', 1],     # 4
                [5, 6, 40, True, 'HS', 1],     # 5
                [5, 3, 48, True, 'HS', 1],     # 6
                [5, 3, 48, True, 'HS', 1],     # 7
                [5, 6, 96, True, 'HS', 2],     # 8
                [5, 6, 96, True, 'HS', 1],     # 9
                [5, 6, 96, True, 'HS', 1],     # 10
            ]
            stem_channels = 16
            stem_stride = 1
            last_channels = 1024

        elif mode == 'large':
            # Configuration for MobileNetV3-Large
            inverted_residual_setting = [
                 k, exp, c, se, act, s
                [3, 1, 16, False, 'RE', 1],    # 0
                [3, 4, 24, False, 'RE', 2],    # 1
                [3, 3, 24, False, 'RE', 1],    # 2
                [5, 3, 40, True, 'RE', 2],     # 3
                [5, 3, 40, True, 'RE', 1],     # 4
                [5, 3, 40, True, 'RE', 1],     # 5
                [3, 6, 80, False, 'HS', 2],    # 6
                [3, 2.5, 80, False, 'HS', 1],  # 7
                [3, 2.3, 80, False, 'HS', 1],  # 8
                [3, 2.3, 80, False, 'HS', 1],  # 9
                [3, 6, 112, True, 'HS', 1],    # 10
                [3, 6, 112, True, 'HS', 1],    # 11
                [5, 6, 160, True, 'HS', 2],    # 12
                [5, 6, 160, True, 'HS', 1],    # 13
                [5, 6, 160, True, 'HS', 1],    # 14
            ]
            stem_channels = 16
            stem_stride = 1
            last_channels = 1280

        elif mode == 'huge':
            # Configuration for (custom) MobileNetV3-Huge
            inverted_residual_setting = [
                # k, exp, c, se, act, s
                [3, 2, 20, False, 'RE', 1],    # 0 
                [3, 8, 30, False, 'RE', 2],    # 1 
                [3, 6, 30, False, 'RE', 1],    # 2 
                [5, 6, 50, True, 'RE', 2],     # 3 
                [5, 6, 50, True, 'RE', 1],     # 4 
                [5, 6, 50, True, 'RE', 1],     # 5 
                [3, 12, 100, False, 'HS', 2],  # 6 
                [3, 5, 100, False, 'HS', 1],   # 7 
                [3, 4.6, 100, False, 'HS', 1], # 8 
                [3, 4.6, 100, False, 'HS', 1], # 9 
                [3, 12, 140, True, 'HS', 1],   # 10 
                [3, 12, 140, True, 'HS', 1],   # 11 
                [5, 12, 200, True, 'HS', 2],   # 12 
                [5, 12, 200, True, 'HS', 1],   # 13 
                [5, 12, 200, True, 'HS', 1],   # 14 
            ]
            stem_channels = 16
            stem_stride = 1
            last_channels = 2048

        else:
            print("No valid model type passed.")
            return
        
        # Scale channels based on width multiplier
        stem_channels = _make_divisible(stem_channels * width_mult, 8)
        
        # Build model stem
        self.stem = ConvBNActivation(3, stem_channels, kernel_size=3, stride=stem_stride, activation=nn.Hardswish)
        
        # Build inverted residual blocks
        features = []
        in_channels = stem_channels
        
        for k, exp, c, se, act, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult, 8)
            activation = nn.ReLU if act == 'RE' else nn.Hardswish
            features.append(
                InvertedResidual(
                    in_channels, 
                    out_channels,
                    kernel_size=k,
                    expand_ratio=exp,
                    stride=s,
                    se_ratio=0.25 if se else 0,
                    activation=activation
                )
            )
            in_channels = out_channels
        
        self.features = nn.Sequential(*features)
        
        # Building last layers
        self.conv_last = ConvBNActivation(in_channels, last_channels, kernel_size=1, activation=nn.Hardswish)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Backbone
        x = self.features(x)
        
        # Pooling
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x):
        x = self.stem(x)  # Stem convolution
        x = self.features(x)  # Inverted residual blocks
        x = self.conv_last(x)  # Final conv layer
        x = self.avgpool(x)  # Global average pooling
        x = torch.flatten(x, 1)  # Flatten to vector
        return x  # Feature vector
    
    def _initialize_weights(self):
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)