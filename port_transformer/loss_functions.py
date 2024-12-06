import torch

from torch.nn.modules.loss import _Loss


class SharpeRatioLoss(_Loss):
    def __init__(self, cost_rate, device="cpu"):
        super(SharpeRatioLoss, self).__init__()
        self.cost_rate = cost_rate
        self.device = device

    def forward(self, pred_weights, asset_returns):
        return sharpe_ratio_loss(pred_weights, asset_returns, self.cost_rate, self.device)


def sharpe_ratio_loss(
    pred_weights: torch.Tensor, 
    asset_returns: torch.Tensor, 
    cost_rate: float,
    device: str = "cpu",
):
    # A Sharpe ratio less than 1 is considered bad. 
    # From 1 to 1.99 is considered adequate/good, from 2 to 2.99 is considered very good,
    # and greater than 3 is considered excellent
    # You can get a negative Sharpe ratio 

    # transaction costs at timestep 0 are zero as we've not changed asset weights
    trans_costs_t0 = torch.zeros((pred_weights.shape[0], 1), device=device)

    # Calculate transaction costs
    trans_costs = torch.cat(
        [
            trans_costs_t0,
            cost_rate
            * torch.sum(
                torch.abs(pred_weights[:, 1:, :] - pred_weights[:, :-1, :]), axis=-1
            ),
        ],
        dim=-1,
    )

    # Calculate portfolio returns
    portfolio_returns = torch.sum(pred_weights * asset_returns, dim=-1) - trans_costs

    # lets compute sharpe ratio on log returns instead

    # log_portfolio_returns = (portfolio_returns + 1).log()

    # avg_log_returns = torch.mean(log_portfolio_returns, dim=-1)
    # volatility = torch.std(log_portfolio_returns, dim=-1)

    # Compute Sharpe ratio
    # sharpe_ratio = avg_log_returns / volatility

    # # Compute Sharpe ratio
    avg_returns = torch.mean(portfolio_returns, dim=-1)
    volatility = torch.std(portfolio_returns, dim=-1)

    # # Sharpe ratios for all items in batch
    sharpe_ratio = avg_returns / volatility

    # log_sharpe_ratio = avg_returns.clamp(min=1e-8).log() - volatility.log()

    # return - log_sharpe_ratio.mean()

    # Since sharpe ratio can be negative let's clamp to some small positive number
    # instead if we're going to take the log. 
    # batch sharpe ratio
    batch_sharpe_ratio = sharpe_ratio.mean()
    # batch_sharpe_ratio = sharpe_ratio.mean().clamp(min=1e-8)

    # # batch log sharpe ratio:
    # log_sharpe_ratio = batch_sharpe_ratio.log()

    return -batch_sharpe_ratio
