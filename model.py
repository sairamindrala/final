import torch
import torch.nn as nn

class LESRCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super(LESRCNN, self).__init__()
        
        # Information Extraction and Enhancement Block (IEEB)
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual Block (RB)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(4)]  # Number of residual blocks
        )
        
        # Information Refinement Block (IRB)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.relu(self.conv1(x))  # IEEB
        out = self.res_blocks(out)      # RB
        out = self.conv2(out)           # IRB part 1
        out = self.conv3(out)           # IRB part 2
        return out

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Adding the residual connection
        return out

# For testing purposes, you can include a small test
if __name__ == '__main__':
    model = LESRCNN()
    print(model)
