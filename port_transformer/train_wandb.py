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
    ConstantLR
)
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
# import mlflow
from functools import partial
import tempfile
import wandb
from utils import save_model

random_seed = 1234

torch.manual_seed(random_seed)
torch.autograd.set_detect_anomaly(True)

wandb_project = "debugging"

debugging_config = "_shared_embs"


# unit_multiplier = 1e4
unit_multiplier = 1

device = "mps" if torch.backends.mps.is_available() else "cuda"

loss_funcs_dict = {"sharpe_ratio_tc=2bp": SharpeRatioLoss(cost_rate=2, device=device)}

save_only_if_better = False
save_freq = 1 # save every x epochs

# figure out what's going on with the initializations

# Have saved normalized returns --> seems to overfit to single batch now 
# (with simplified debugging architecture), but very wild fluctuations at the beginning!

# TODO: Let's continue trying to overfit to a single batch just trying to predict the target sequence 
# and just using MSE loss. 

# Add each component back in one by one and see if anything breaks
# Last thing to add will be sharpe ratio loss

# Next:
# inspect T2V layer, are weights and gradients as we expect?
# Then try adding grn back in next

# Try RMS normalization?

# do we need any extra regularisation here? 
# should we look at weight initialisation? LSUV init, so dont need to worry 
# about correct initialisation for whatever activation functions we're using in the GRN

# add scheduler back in

parameters = {
    "n_epochs": 50,
    "d_model": 128,
    "nhead": 8,
    "num_layers": 1,
    "d_grn": 512,
    "window": 50,
    "tau": 25,
    "max_lr": 0.01,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "shuffle_batches": True,
    "use_src_causal_mask": False,
    "norm_first": True,
    "loss_function": "sharpe_ratio_tc=2bp",
    "dropout": 0.1,
}

DEBUG = False


data_path = Path("/mnt/c/Users/hmcp2/hugo_repos/portfolio-wizard/portfolio_ai/data")
# df = pd.read_csv(data_path / "all_etfs_returns.csv")
# df = pd.read_csv(data_path / "all_etfs_log_returns.csv")
df = pd.read_csv(data_path / "all_etfs_returns_norm.csv")


train_df = df[df["Date"] <= "2015-12-31"]
val_size = int(0.1 * len(train_df))

test_df = df[(df["Date"] >= "2016-01-01") & (df["Date"] <= "2016-12-31")]

# Asset returns

asset_names = df.columns[1:]
asset_returns = torch.tensor(train_df[asset_names].values*unit_multiplier).float()

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





# print(pt_model)


model_parameters = filter(lambda p: p.requires_grad, pt_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

run_name = f"PT_{int(params/1000000)}M{debugging_config}"


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
    train_samples[:8], # let's overfit to single batch first
    batch_size=parameters["train_batch_size"],
    shuffle=parameters["shuffle_batches"],
)
val_loader = DataLoader(
    val_samples[:8], # let's overfit to single batch first
    batch_size=parameters["val_batch_size"],
    shuffle=parameters["shuffle_batches"],
)


# Define optimizer
optimizer = Adam(
    pt_model.parameters(),
    lr=parameters["max_lr"],
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0, # ? This we should try with weight decay too. Not for layer norm and biases tho.
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


# wandb.log_artifact()




# exp = mlflow.get_experiment_by_name(experiment_name)

# if not exp:
#     exp_id = mlflow.create_experiment(experiment_name)
# else:
#     exp_id = exp.experiment_id

modes = ["training", "validation"]

num_steps = parameters["n_epochs"] * len(train_loader) + parameters["n_epochs"] * len(
    val_loader
)

pbar_steps = tqdm(total=num_steps, desc="Total progess")

if not DEBUG:
    wandb.init(
        project=wandb_project,
        # run_name=run_name,
        name=run_name,
        config=parameters,
    )
    wandb.watch(pt_model, log='all', log_graph=True, log_freq=1)

# with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
#     run_id = mlflow.active_run().info.run_id

#     mlflow.log_params(parameters)
    
from torch.nn import MSELoss
debug_loss_func = MSELoss()


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
            # if idx > 0: continue # let's overfit to single batch first
            if mode == "training":
                optimizer.zero_grad()

            # split asset returns into src and tgt sequences 

            # src sequence, 0 -> tau 
            src = batch[:, : -parameters["tau"], :].to(device)

            # tgt sequence for decoder input, asset returns for tau+1 -> t
            tgt_dec_input = batch[:, -parameters["tau"] :-1, :].to(device)


            # get asset weights for tau+1 -> t 
            pred_weights = pt_model(
                src=src,
                tgt=tgt_dec_input,
            )

            # tgt sequence for loss calculation, asset returns for tau+2 -> t+1
            tgt_returns = batch[:, -parameters["tau"] +1 :, :].to(device)

            debug_loss = debug_loss_func(pred_weights, tgt_returns)

            batch_loss = debug_loss
            
            # batch_loss = loss_funcs_dict[parameters["loss_function"]](
            #     pred_weights=pred_weights, asset_returns=tgt_returns
            # )
   
            wandb.log({f"{mode}_batch_loss": batch_loss.item()})

            # mlflow.log_metric(
            #     f"{mode}_batch_loss", batch_loss, step=idx + epoch * num_steps
            # )

            if mode == "training":
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
            losses_dict[mode].append(batch_loss.item())

            pbar_steps.update(1)

    train_loss = np.mean(losses_dict["training"])
    val_loss = np.mean(losses_dict["validation"])

    if not DEBUG:
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        for name, param in pt_model.named_parameters():
            mean = param.data.mean()
            std = param.data.std()
            wandb.log({f"{name}_mean": param.data.mean(), f"{name}_std": param.data.std()})
            # print(f"{name} - Mean: {mean}, Std: {std}")

    # mlflow.log_metric("train_loss", train_loss, step=epoch)
    # mlflow.log_metric("val_loss", val_loss, step=epoch)

    # print(f"Epoch {epoch}: Train Loss {train_loss}, Validation Loss {val_loss}")
    pbar_steps.write(f"Epoch {epoch}: Train Loss {train_loss}, Validation Loss {val_loss}")

    # if not DEBUG:
    #     if save_only_if_better:
    #         if val_loss < prev_val_loss:
    #             save_model(pt_model, optimizer, epoch)
    #             prev_val_loss = val_loss
    #     elif epoch % save_freq == 0 or epoch == parameters["n_epochs"]:
    #         save_model(pt_model, optimizer, epoch)
if not DEBUG:
    wandb.finish()
# to load model again:
# checkpoint = torch.load('models/model_42.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
