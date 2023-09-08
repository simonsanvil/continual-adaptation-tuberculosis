from bacili_detection.configs.loss_pred import DETRLossPredictionExperimentConfig


backbone_pred_config = DETRLossPredictionExperimentConfig(
    batch_size=2,
    num_epochs=90,
    stop_patience=7,
    lrs_patience=3,
    eval_every=5,
    device="cpu",
    target_loss="loss_ce",
    model_save_dir="bacili_detection/active_learning/strategies/loss_prediction/models/detr/",
    lr=0.001, log_wandb=True,
    wandb_project="bacilli-detection"
)