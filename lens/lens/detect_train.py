import pandas as pd
from pytorch_lightning import Trainer

import ast
import numpy as np

from lens.models.detect_metric import DetectMetric
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


# Optional: set seed or deterministic behavior
from pytorch_lightning import seed_everything
seed_everything(42)

# create these files through Dataset_Generation.ipynb
# Note: additional features are not used in final DETECT and not added to the dataset, but the repository supports making a custom neural network with more features.
train_path = ["lens/data/SimpEvalDE_train_train.csv.csv"]
val_path = ["lens/data/SimpEvalDE_train_val.csv"]

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss/dataloader_idx_1",
    mode="min",       # or "min" if you're monitoring val_loss
    patience=3,       # stop after 2 epochs with no improvement
    verbose=True,
)

def run_model_iteration(parameters_in, run_name):
    #1. Default LENS - no change in parameters
    model = DetectMetric(prediction_type="single", add_features=False, alpha=1, beta=0)


    ## Define your model
    model = DetectMetric(
        train_data=train_path,
        validation_data=val_path,
        topk=1,                    # LENS-style loss (top-k references)
        batch_size = 4,
        encoder_model = "RoBERTa",
        encoder_learning_rate = 1e-5,
        #learning_rate = 2e-5,
        learning_rate= 1e-05,
        #dropout = 0.3
        #pretrained_model= "benjamin/roberta-base-wechsel-german"
    )

    wandb_logger = WandbLogger(
        project="DETECT-Final-training",
        log_model=True,
        config=model.hparams,
        name = run_name 
    )

  
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss/dataloader_idx_1",  # or "val_spearman", etc.
    mode="min",           # "min" for loss, "max" for correlation
    save_top_k=1,
    save_weights_only=False,  # Saves full model (recommended)
    dirpath="checkpoints/",
    filename=f"best-{run_name}-{{epoch:02d}}",
    )


    # Initialize the PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=5,
        accelerator="auto",       # uses GPU if available
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],  # ← ✅ add this

    )

    # Start training
    trainer.fit(model)


    

def main():
   
   runs = [

    ({"prediction_type": "single", "add_features": False, "alpha": 0.5, "beta": 0.5,
      "dropout": 0.3, "learning_rate": 1e-05},
     "LENS_multi_dropout03_lr1e05_MSE05"),

     ]
   
   for parameters_in, run_name in runs:
    print(f"\n--- Running {run_name} ---")
    run_model_iteration(parameters_in, run_name)


if __name__ == "__main__":
    main()
