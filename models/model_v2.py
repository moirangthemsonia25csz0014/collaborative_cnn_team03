import torch
import torch.nn as nn
from torchvision import models


class PlantDiseaseResNet18(nn.Module):

    def __init__(self, num_classes=15, pretrained=False, hidden_dim=512, dropout=0.5, attn_reduction=16):
        super(PlantDiseaseResNet18, self).__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # --- Channel attention on backbone features (SE-style) ---
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // attn_reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // attn_reduction, in_features, bias=True),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes, bias=True)
        )

        # Init classifier + attention
        for m in list(self.classifier.modules()) + list(self.attention.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = self.backbone(x)       # (B, 512)
        attn_weights = self.attention(features)  # (B, 512) in [0,1]
        features = features * attn_weights       # channel-wise reweighting
        out = self.classifier(features)
        return out

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
