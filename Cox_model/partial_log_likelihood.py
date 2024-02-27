# https://github.com/havakv/pycox/blob/master/pycox/models/cox.py

import torchtuples as tt
import pandas as pd
import numpy as np

class likelihood():
    duration_col = 'duration'
    event_col = 'event'
    
    def target_to_df(self, target):
        durations, events = tt.tuplefy(target).to_numpy()
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events}) 
        return df
    
    
    def partial_log_likelihood(self, target, g_preds=None, batch_size=8224, eps=1e-7, eval_=True,
                               num_workers=0):
        '''Calculate the partial log-likelihood for the events in datafram df.
        This likelihood does not sample the controls.
        Note that censored data (non events) does not have a partial log-likelihood.

        Arguments:
            input {tuple, np.ndarray, or torch.tensor} -- Input to net.
            target {tuple, np.ndarray, or torch.tensor} -- Target labels.

        Keyword Arguments:
            g_preds {np.array} -- Predictions from `model.predict` (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            Partial log-likelihood.
        '''
        df = self.target_to_df(target)
        return (df
                .assign(_g_preds=g_preds)
                .sort_values(self.duration_col, ascending=False)
                .assign(_cum_exp_g=(lambda x: x['_g_preds']
                                    .pipe(np.exp)
                                    .cumsum()
                                    .groupby(x[self.duration_col])
                                    .transform('max')))
                .loc[lambda x: x[self.event_col] == 1]
                .assign(pll=lambda x: x['_g_preds'] - np.log(x['_cum_exp_g'] + eps))
                ['pll'])