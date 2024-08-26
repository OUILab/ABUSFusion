import torch
import torch.nn as nn


class MSECorrelationLoss(nn.Module):
    def __init__(self, lambda_corr=0.5):
        super(MSECorrelationLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_corr = lambda_corr

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)

        # Compute correlation loss
        pred_mean = predictions.mean(dim=0, keepdim=True)
        target_mean = targets.mean(dim=0, keepdim=True)
        pred_std = predictions.std(dim=0, unbiased=False)
        target_std = targets.std(dim=0, unbiased=False)

        correlation = ((predictions - pred_mean) * (targets - target_mean)).mean(
            dim=0
        ) / (pred_std * target_std + 1e-8)
        corr_loss = 1 - correlation.mean()

        return mse_loss + self.lambda_corr * corr_loss
