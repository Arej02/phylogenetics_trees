import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

@torch.no_grad()
def evaluate(model, X_tensor, y_true_transformed, target_names):
    device = next(model.parameters()).device
    model.eval()
    
    pred_transformed = model(X_tensor.to(device)).cpu().numpy()

    pred_orig = np.expm1(pred_transformed)
    true_orig = np.expm1(y_true_transformed)

    mae = mean_absolute_error(true_orig, pred_orig, multioutput='raw_values')
    r2  = r2_score(true_orig, pred_orig, multioutput='uniform_average')

    return mae, r2, pred_orig, true_orig