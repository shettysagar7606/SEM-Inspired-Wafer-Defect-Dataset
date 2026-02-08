"Baseline CNN architecture for future training"
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaferDefectCNN(nn.Module):
    """
    Baseline CNN for SEM-inspired grayscale wafer defect images.
    Phase-1: Architecture definition only.
    """

    def __init__(self, num_classes=6):
        super(WaferDefectCNN, self).__init__()

        # -------- Feature Extraction --------
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # -------- Classification --------
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (B, 1, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 32, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))  # -> (B, 64, 16, 16)

        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # Sanity check
    model = WaferDefectCNN(num_classes=6)
    dummy_input = torch.randn(1, 1, 128, 128)
    output = model(dummy_input)
    print("Output shape:", output.shape)
