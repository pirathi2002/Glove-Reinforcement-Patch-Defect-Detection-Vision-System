# anomaly_model.py - autoencoder / anomaly logic
import os
import torch
from anomalib.models import Fastflow
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint
from src.Data_module import create_glove_datamodule


def train_fastflow(
    dataset_root="data",
    image_size=224,
    batch_size=16,
    max_epochs=20,
    accelerator="auto",
    devices=1,
):
    """
    Train FastFlow model for glove patch defect detection
    """

    # Create datamodule
    datamodule = create_glove_datamodule(
        dataset_root=dataset_root,
        image_size=image_size,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )
    datamodule.setup()

    # Create FastFlow model
    model = Fastflow(
        input_size=(image_size, image_size),
        backbone="resnet18",
        pre_trained=True,
    )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="image_AUROC",
        mode="max",
        save_last=True,
        save_top_k=1,
    )

    # Training engine
    engine = Engine(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
    )

    #  Train model
    engine.fit(model=model, datamodule=datamodule)

    # Test model
    engine.test(model=model, datamodule=datamodule)

    print("FastFlow training completed")


if __name__ == "__main__":
    train_fastflow()

