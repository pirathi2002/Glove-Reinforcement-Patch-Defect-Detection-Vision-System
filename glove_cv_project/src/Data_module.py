from anomalib.data import Folder
from pathlib import Path


def create_glove_datamodule(
    dataset_root: str = "data",
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    num_workers: int = 4,
):
    """
    Custom Folder DataModule for Glove Reinforced Patch Defect Detection
    """

    dataset_root = Path(dataset_root)

    datamodule = Folder(
        name="glove_patch_defect",
        root=dataset_root,

        # NORMAL images
        normal_dir="train/acceptable",

        # ABNORMAL images
        abnormal_dir=[
            "test/unacceptable",
        ],

        extensions=(".jpg", ".png", ".jpeg"),

        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    return datamodule


