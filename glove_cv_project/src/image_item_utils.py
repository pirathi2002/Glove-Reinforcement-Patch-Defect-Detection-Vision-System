import torch
from anomalib.data.dataclasses.torch import ImageItem
import cv2


def create_image_item(image_path: str):
    """
    Convert image into Anomalib ImageItem
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0

    item = ImageItem(
        image=img_tensor,
        image_path=image_path
    )

    return item
