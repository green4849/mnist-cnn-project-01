import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Inverted Residual Block 클래스 정의 (MobileNetV2 아이디어)
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=4):
        """
        in_channels: 입력 채널
        out_channels: 출력 채널
        stride: 1 또는 2 (다운샘플링 여부)
        expansion_factor: 채널을 몇 배로 '팽창'시킬지 (t)
        """
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor

        # 1. Pointwise (Expansion)
        self.pw1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 2. Depthwise Convolution
        self.dw = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 3. Pointwise (Projection)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # 4. Shortcut connection (stride=1이고 채널이 같을 때만 사용)
        self.use_shortcut = (self.stride == 1 and in_channels == out_channels)

    def forward(self, x):
        identity = x
        
        # ReLU6는 ReLU보다 낮은 비선형성을 가져 경량 모델에 유리합니다.
        out = F.relu6(self.bn1(self.pw1(x)))
        out = F.relu6(self.bn2(self.dw(out)))
        out = self.bn3(self.pw2(out)) # 마지막엔 ReLU 없음

        if self.use_shortcut:
            out = out + identity # Residual Connection
            
        return out


# 2. 메인 CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Stem (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(1, 24, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 2. Body (Inverted Residual Blocks Stack)
        # Block 1: 14x14 -> 7x7 (Downsampling)
        self.block1 = InvertedResidualBlock(in_channels=24, out_channels=32, stride=2, expansion_factor=4)
        
        # Block 2: 7x7 -> 7x7 (Identity)
        self.block2 = InvertedResidualBlock(in_channels=32, out_channels=48, stride=1, expansion_factor=4)
        
        # Block 3: 7x7 -> 7x7 (Identity, t=3으로 줄여 파라미터 맞춤)
        self.block3 = InvertedResidualBlock(in_channels=48, out_channels=48, stride=1, expansion_factor=3)
        
        # 3. Classifier Head (마지막 채널 수 48)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, 10) # 48*10 + 10 = 490
        self.dropout = nn.Dropout(0.25) # 98.10% 달성 시의 값

    def forward(self, x):
        x = x.view(-1, 1, 28, 28) # 784 벡터를 28x28 이미지로
        
        # Stem
        x = self.pool1(F.relu6(self.bn1(self.conv1(x))))
        
        # Body
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Classifier
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 3. 파라미터 수 검증용 코드
if __name__ == "__main__":
    batch_size = 32
    in_dim = 784
    x = torch.randn(batch_size, in_dim)
    
    model = CNN()
    output = model(x)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: InvertedResidual_3-Block (t=4,4,3)")
    print(f"Number of trainable parameters: {num_params:,}") # 약 34,346개

    print(f"Input shape (from train script):  {x.shape}")
    print(f"Output shape (to loss function): {output.shape}")

    try:
        from thop import profile
        macs_input = torch.randn(1, 784)
        macs, params = profile(model, inputs=(macs_input, ))
        print(f'MACs: {macs / 1e9:.5f} G, Params: {params / 1e6:.5f} M')
    except Exception as e:
        print(f"Could not calculate MACs. {e}")