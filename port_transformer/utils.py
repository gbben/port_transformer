import tempfile
import wandb
import torch

def save_model(pt_model, optimizer, epoch):
    with tempfile.TemporaryDirectory() as tmp:
        filepath = f"{tmp}/pt_ckpt_{epoch}.pth"
        # save weights
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": pt_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            filepath,
        )
        wandb.log_artifact(filepath, "model_and_optim_state_dicts", type="model")