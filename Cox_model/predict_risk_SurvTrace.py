# https://github.com/RyanWangZf/SurvTRACE/blob/main/survtrace/model.py

import torch.nn.functional as F
import torch
import pandas as pd

def pad_col(hazard, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
#     if len(hazard.shape) != 2:
#         raise ValueError(f"Only works for `phi` tensor that is 2-D.")
#     pad = torch.zeros_like(hazard[:, :1])
    pad = torch.zeros_like(hazard)
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([hazard, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, hazard], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

class Predict():
    def predict_hazard(self, preds, batch_size=None):
        hazard = F.softplus(preds)
        hazard = pad_col(hazard, where="start")
        return hazard
    
    def predict_risk(self, preds, batch_size=None):
        surv = self.predict_surv(preds, batch_size)
        return 1- surv

    def predict_surv(self, preds, batch_size=None, epsilon=1e-7):
        hazard = self.predict_hazard(preds, batch_size)
        # surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        surv = hazard.cumsum(1).mul(-1).exp()
        return surv
    
    def predict_surv_df(self, preds, batch_size=None):
        surv = self.predict_surv(preds, batch_size)
        return pd.DataFrame(surv.to("cpu").numpy().T)
#         return pd.DataFrame(surv.to("cpu").numpy().T, self.duration_index)