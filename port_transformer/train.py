import pandas as pd
import torch
from pathlib import Path
from modules import PortfolioTransformer
from loss_functions import SharpeRatioLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    StepLR,
    OneCycleLR,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
)
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import mlflow
from functools import partial
import tempfile

device = "mps" if torch.backends.mps.is_available() else "cuda"


loss_funcs_dict = {"sharpe_ratio_tc=2bp": SharpeRatioLoss(cost_rate=0.0002, device=device)}

save_only_if_better = True
save_freq = 5


parameters = {
    "n_epochs": 10,
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "d_grn": 1024,
    "window": 50,
    "tau": 25,
    "max_lr": 0.001,
    "train_batch_size": 16,
    "val_batch_size": 16,
    "suffle_batches": True,
    "use_src_causal_mask": False,
    "norm_first": False,
    "loss_function": "sharpe_ratio_tc=2bp",
    "dropout": 0.1,
}

experiment_name = "pt_train_2015"

data_path = Path("../data")
df = pd.read_csv(data_path / "all_etfs_returns.csv")


train_df = df[df["Date"] <= "2015-12-31"]
val_size = int(0.1 * len(train_df))

test_df = df[(df["Date"] >= "2016-01-01") & (df["Date"] <= "2016-12-31")]

# Asset returns

asset_names = df.columns[1:]
asset_returns = torch.tensor(train_df[asset_names].values).float()

n_assets = asset_returns.shape[-1]

train_asset_returns = asset_returns[:-val_size, :]
val_asset_returns = asset_returns[-val_size:, :]


pt_model = PortfolioTransformer(
    n_assets=n_assets,
    d_model=parameters["d_model"],
    nhead=parameters["nhead"],
    num_layers=parameters["num_layers"],
    d_grn=parameters["d_grn"],
    use_src_causal_mask=parameters["use_src_causal_mask"],
    device=device,
    norm_first=parameters["norm_first"],
    dropout=parameters["dropout"],
)

model_parameters = filter(lambda p: p.requires_grad, pt_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

run_name = f"portfolio_transformer_{int(params/1000000)}M"


train_samples = []
for idx in range(len(train_asset_returns) - parameters["window"]):
    start = idx
    end = idx + parameters["window"]
    train_samples.append(train_asset_returns[start:end, :])


val_samples = []
for idx in range(len(val_asset_returns) - parameters["window"]):
    start = idx
    end = idx + parameters["window"]
    val_samples.append(val_asset_returns[start:end, :])

train_loader = DataLoader(
    train_samples,
    batch_size=parameters["train_batch_size"],
    shuffle=parameters["suffle_batches"],
)
val_loader = DataLoader(
    val_samples,
    batch_size=parameters["val_batch_size"],
    shuffle=parameters["suffle_batches"],
)

# Define optimizer
optimizer = Adam(
    pt_model.parameters(),
    lr=parameters["max_lr"],
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
)

# try other schedulers
# scheduler = StepLR(optimizer=optimizer, step_size=4, gamma=0.001)
# scheduler = OneCycleLR(
#     optimizer,
#     max_lr=parameters[
#         "max_lr"
#     ],  # Upper learning rate boundaries in the cycle for each parameter group
#     steps_per_epoch=len(train_loader),  # The number of steps per epoch to train for.
#     epochs=parameters["n_epochs"],  # The number of epochs to train for.
#     anneal_strategy="cos",
# )  # Specifies the annealing strategy
# linear warmup decay
scheduler = CosineAnnealingLR(
    optimizer=optimizer, T_max=parameters["n_epochs"] * len(train_loader)
)

parameters["optimizer"] = optimizer
parameters["scheduler"] = scheduler


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
        mlflow.log_artifact(filepath, "model_weights")


exp = mlflow.get_experiment_by_name(experiment_name)

if not exp:
    exp_id = mlflow.create_experiment(experiment_name)
else:
    exp_id = exp.experiment_id

modes = ["training", "validation"]

num_steps = parameters["n_epochs"] * len(train_loader) + parameters["n_epochs"] * len(
    val_loader
)

pbar_steps = tqdm(total=num_steps, desc="Total progess")

with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
    run_id = mlflow.active_run().info.run_id

    mlflow.log_params(parameters)
    prev_val_loss = 1000
    for epoch in range(parameters["n_epochs"]):
        losses_dict = {"training": [], "validation": []}

        for mode, data_loader in zip(modes, [train_loader, val_loader]):
            if mode == "training":
                pt_model.train()
            else:
                pt_model.eval()

            # num_steps = len(data_loader)
            # pbar_steps = tqdm(total=num_steps, desc=f"Epoch {epoch} {mode} progess")
            for idx, batch in enumerate(data_loader):
                if mode == "training":
                    optimizer.zero_grad()

                # split asset returns into src and tgt sequences
                src = batch[:, : -parameters["tau"], :].to(device)
                tgt = batch[:, -parameters["tau"] :, :].to(device)

                pred_weights = pt_model.forward(
                    src=src,
                    tgt=tgt,
                )
                batch_loss = loss_funcs_dict[parameters["loss_function"]](
                    pred_weights=pred_weights, asset_returns=tgt
                )
                losses_dict[mode].append(batch_loss.item())

                # mlflow.log_metric(
                #     f"{mode}_batch_loss", batch_loss, step=idx + epoch * num_steps
                # )

                if mode == "training":
                    batch_loss.backward()
                    optimizer.step()
                pbar_steps.update(1)

        train_loss = np.mean(losses_dict["training"])
        val_loss = np.mean(losses_dict["validation"])

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # print(f"Epoch {epoch}: Train Loss {train_loss}, Validation Loss {val_loss}")
        pbar_steps.write(f"Epoch {epoch}: Train Loss {train_loss}, Validation Loss {val_loss}")

        if save_only_if_better:
            if val_loss < prev_val_loss:
                save_model(pt_model, optimizer, epoch)
                prev_val_loss = val_loss
        elif epoch % save_freq == 0 or epoch == parameters["n_epochs"]:
            save_model(pt_model, optimizer, epoch)


# to load model again:
# checkpoint = torch.load('models/model_42.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
