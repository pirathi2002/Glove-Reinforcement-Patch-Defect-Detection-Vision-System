# heatmap.py - difference -> heatmap
# heatmap.py
import os
import torch
from anomalib.visualization import ImageVisualizer
from anomalib.visualization.image.item_visualizer import visualize_image_item
from anomalib.data import ImageItem

class HeatmapGenerator:
    """
    Generates anomaly heatmaps for glove patch defects using Anomalib Visualizer
    """
    def __init__(self, model, save_dir="c:/glove patch project_pirathi/glove_cv_project/logs/heatmaps"):
        self.model = model
        self.model.eval()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.visualizer = ImageVisualizer(
            fields_config={
                "image": {},
                "anomaly_map": {"colormap": True, "normalize": True},
                "pred_mask": {"mode": "fill", "color": (255, 0, 0), "alpha": 0.5},
                "gt_mask": {"mode": "contour", "color": (0, 255, 0), "alpha": 0.7}
            }
        )

    @torch.no_grad()
    def generate(self, image_tensor, image_path=None):
        """
        Generate heatmap and overlay for a single image
        """
        # Prediction
        output = self.model.predict(image_tensor.unsqueeze(0))
        anomaly_map = output.get("anomaly_map")
        anomaly_score = output.get("anomaly_score").item()
        
        # want to set threshold based on validation set later
        pred_label = "Abnormal" if anomaly_score > 0.5 else "Normal"

        # Create ImageItem for visualization
        item = ImageItem(
            image_path=image_path,
            image=image_tensor,
            anomaly_map=anomaly_map
        )

        # Generate visualization
        vis_result = visualize_image_item(
            item,
            fields=["image", "anomaly_map"],
            fields_config={"anomaly_map": {"colormap": True, "normalize": True}}
        )

        # Save visualization locally
        if image_path is not None:
            save_path = os.path.join(self.save_dir, f"heatmap_{os.path.basename(image_path)}")
            vis_result.save(save_path)

        return pred_label, anomaly_score, vis_result.get_image()  # Returns label, score, numpy overlay
