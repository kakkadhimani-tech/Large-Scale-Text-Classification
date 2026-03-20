import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn


class CustomTextClassifier(nn.Module):
    def __init__(self, input_dim=20000, num_classes=24):
        super(CustomTextClassifier, self). __init__()
        # Layer 1: Bada dimaag (20k features -> 1024 neurons)
        self.layer1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024) # Speed badhane ke liye
        
        # Layer 2: Deep layer (1024 -> 512 neurons)
        self.layer2 = nn.Linear(1024, 512)
        
        # Layer 3: Final Layer (512 -> 24 classes)
        self.output_layer = nn.Linear(512, num_classes)
        
        # Dropout: 30% neurons ko rest dega taaki overfitting na ho
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    # Test block
    model = CustomTextClassifier(input_dim=20000, num_classes=24)
    print(" Upgraded Model Architecture Created!")
    print(f" Total Params: {sum(p.numel() for p in model.parameters()):,}")