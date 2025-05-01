# export_to_onnx.py

import torch
import torch.nn as nn

# ------------------------------------------------
# 1) MODEL DEFINITION (no training code here)
# ------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------
# 2) EXPORT LOGIC
# ------------------------------------------------
def main():
    # a) instantiate & switch to eval mode
    model = SmallCNN().eval()

    # b) load your checkpoint
    ckpt = torch.load("/Users/baveethora/Desktop/ Velox/meko_use_karo.pth", map_location="cpu")
    # if you saved a dict with "state_dict":
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # assume ckpt itself is a state_dict or a full model
        try:
            model.load_state_dict(ckpt)
        except RuntimeError:
            model = ckpt
            model.eval()

    # c) dummy input matching your training size (3×128×128)
    dummy = torch.randn(1, 3, 128, 128, dtype=torch.float32)

    # d) export to ONNX
    torch.onnx.export(
        model,
        dummy,
        "smallcnn.onnx",
        input_names=["image"],
        output_names=["score"],
        opset_version=11,
        dynamic_axes={
            "image": {0: "batch"},
            "score": {0: "batch"},
        },
    )
    print("✅ Exported smallcnn.onnx")

if __name__ == "__main__":
    main()
