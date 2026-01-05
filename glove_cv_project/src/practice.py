# src/practice.py

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from types import SimpleNamespace

from anomalib.models import Fastflow
from anomalib.engine import Engine
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.data import ImageItem


#  Custom Dataset for practice

class GloveDataset(Dataset):
    def __init__(self, folder, max_items=None):
        self.image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png"))
        ]
        # Optionally limit the dataset to the first `max_items` files
        if max_items is not None:
            self.image_files = self.image_files[:max_items]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to 256x256 to match model expected feature map shapes
        img_resized = cv2.resize(img, (256, 256))
        img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
        return img_tensor, img_path



#  Load datasets

train_folder = os.path.join(os.path.dirname(__file__), "..", "pictures")
test_folder = os.path.join(os.path.dirname(__file__), "..", "roi_output")

train_dataset = GloveDataset(train_folder, max_items=5)  # limit to first 5 images
test_dataset = GloveDataset(test_folder)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



#  Initialize FastFlow model (robust to download/cert errors)

print('>>> Initializing FastFlow (attempt pretrained=True)')
try:
    model = Fastflow(
        backbone="resnet18",
        pre_trained=True,
        flow_steps=8
    )
    print('>>> FastFlow pretrained init succeeded')
except Exception as e:
    print("Warning: could not initialize pretrained model (will use untrained model). Error:\n", e)
    print('>>> Falling back to FastFlow(pre_trained=False)')
    model = Fastflow(
        backbone="resnet18",
        pre_trained=False,
        flow_steps=8
    )
model.eval()


#  Train FastFlow on 5 images for 10 epochs (enabled)

DO_TRAIN = True
if DO_TRAIN:
    print('>>> Starting training: 5 images, 10 epochs')
    engine = Engine(max_epochs=10)
    # The Engine.fit may try to create symlinks on Windows which requires admin privileges.
    # Try Engine.fit first; if it fails with a symlink OSError, fall back to a minimal manual training loop
    try:
        # The Engine.fit API expects positional dataloaders in some versions
        try:
            engine.fit(model, train_loader)
        except TypeError:
            engine.fit(model=model, train_dataloader=train_loader)
        print('>>> Training finished (Engine.fit)')
    except OSError as e:
        # Typical Windows symlink permission error
        print("Warning: Engine.fit failed due to OS permissions (symlink). Falling back to manual training loop. Error:\n", e)
        # Manual training loop (no experiment logging)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        num_epochs = 1
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            batch_count = 0
            for batch_idx, (imgs, _) in enumerate(train_loader):
                imgs = imgs  # (B, C, H, W)
                # model.training_step expects an object with attribute `image` (not a dict)
                batch = SimpleNamespace(image=imgs)
                try:
                    out = model.training_step(batch, batch_idx)
                except Exception as e2:
                    print("Manual training fallback failed:", e2)
                    raise RuntimeError("Manual fallback failed; please run the script with admin privileges or set DO_TRAIN=False.") from e2
                # Extract loss from output
                if isinstance(out, dict):
                    loss = out.get("loss")
                    if loss is None:
                        # pick first tensor-like value
                        found = [v for v in out.values() if isinstance(v, torch.Tensor)]
                        if found:
                            loss = found[0]
                        else:
                            raise RuntimeError("Could not find loss in training_step output")
                elif isinstance(out, torch.Tensor):
                    loss = out
                else:
                    raise RuntimeError("Unexpected training_step return type: " + str(type(out)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                batch_count += 1
            avg_loss = total_loss / batch_count if batch_count else 0.0
            print(f"Manual training epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}")
        print('>>> Manual training finished')


#  Test and generate heatmaps (saved to logs/heatmaps)

heatmap_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "heatmaps")
os.makedirs(heatmap_dir, exist_ok=True)
model.eval()
for img_tensor, img_path in test_loader:
    # img_tensor: (1, C, H, W) -> squeeze batch
    img_tensor = img_tensor.squeeze(0)
    image_path = img_path[0] if isinstance(img_path, (list, tuple)) else img_path

    with torch.no_grad():
        # Robust prediction: prefer model.predict, else try model(...)
        try:
            output = model.predict(img_tensor.unsqueeze(0))
        except Exception:
            try:
                out = model(img_tensor.unsqueeze(0))
                if isinstance(out, dict):
                    output = out
                elif isinstance(out, (tuple, list)) and len(out) >= 2:
                    output = {"anomaly_map": out[0], "anomaly_score": out[1]}
                else:
                    print("Warning: unexpected model output; using placeholders.")
                    output = {"anomaly_map": torch.zeros((1, 224, 224), dtype=torch.float32), "anomaly_score": torch.tensor(0.0)}
            except Exception as e:
                raise RuntimeError("Model prediction failed: " + str(e))

        anomaly_map = output.get("anomaly_map")
        anomaly_score_tensor = output.get("anomaly_score", 0.0)
        anomaly_score = float(anomaly_score_tensor.item()) if hasattr(anomaly_score_tensor, "item") else float(anomaly_score_tensor)
        pred_label = "Abnormal" if anomaly_score > 0.5 else "Normal"

        # Normalize anomaly_map to torch.Tensor [H, W]
        if isinstance(anomaly_map, np.ndarray):
            anomaly_map = torch.from_numpy(anomaly_map).float()
        if isinstance(anomaly_map, torch.Tensor):
            # If batch or channel dims present, try to reduce to [H, W]
            if anomaly_map.dim() == 4:  # (B, C, H, W)
                anomaly_map = anomaly_map[0]
            if anomaly_map.dim() == 3:  # (C, H, W) or (1, H, W)
                if anomaly_map.shape[0] == 1:
                    anomaly_map = anomaly_map.squeeze(0)
                else:
                    anomaly_map = anomaly_map.mean(dim=0)
            if anomaly_map.dim() == 0:
                # fallback: create zeros matching input image size
                h, w = img_tensor.shape[1], img_tensor.shape[2]
                anomaly_map = torch.zeros((h, w), dtype=torch.float32)
        else:
            # fallback placeholder
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            anomaly_map = torch.zeros((h, w), dtype=torch.float32)

        # Build ImageItem
        item = ImageItem(
            image_path=image_path,
            image=img_tensor,
            anomaly_map=anomaly_map
        )

        # Visualization
        vis = visualize_image_item(
            item,
            fields=["image", "anomaly_map"],
            fields_config={"anomaly_map": {"colormap": True, "normalize": True}}
        )
        try:
            overlay_image = vis.get_image()
        except AttributeError:
            overlay_image = np.asarray(vis)

    # Ensure numpy uint8 RGB for saving
    img_arr = overlay_image if isinstance(overlay_image, np.ndarray) else np.asarray(overlay_image)
    if img_arr.dtype != np.uint8:
        # Normalize floats in [0,1] or scale 0-255
        if img_arr.max() <= 1.0:
            img_arr = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)
        else:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    # Save heatmap (convert RGB->BGR for OpenCV)
    save_name = os.path.splitext(os.path.basename(image_path))[0] + "_heatmap.png"
    save_path = os.path.join(heatmap_dir, save_name)
    try:
        save_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    except Exception:
        save_bgr = img_arr
    cv2.imwrite(save_path, save_bgr)

    # Also save anomaly map as numpy for inspection
    am_save_path = os.path.join(heatmap_dir, os.path.splitext(os.path.basename(image_path))[0] + "_anomaly_map.npy")
    np.save(am_save_path, anomaly_map.detach().cpu().numpy())

    print(f"Saved heatmap: {save_path}  (score: {anomaly_score:.3f})")

