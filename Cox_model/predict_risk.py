# https://github.com/havakv/pycox/blob/master/pycox/models/cox.py
# https://github.com/havakv/pycox/blob/master/pycox/models/cox.py

import numpy as np
import torchtuples as tt
import pandas as pd

class Predict():
    duration_col = 'duration'
    event_col = 'event'
    
    def target_to_df(self, target):
        durations, events = tt.tuplefy(target).to_numpy()
        df = pd.DataFrame({self.duration_col: durations, self.event_col: events}) 
        return df
    
    
    def predict_surv_df(self, preds, target, max_duration=None, batch_size=8224, verbose=False, baseline_hazards_=None,
                        eval_=True, num_workers=0):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require computed baseline hazards.

        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.

        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})

        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(preds, self.target_to_df(target), max_duration, batch_size,
                                                              eval_=eval_, num_workers=num_workers)
        
        return np.exp(-self.predict_cumulative_hazards(preds, max_duration, batch_size, verbose, baseline_hazards_,
                                                       eval_, num_workers))
    
    
    def predict_cumulative_hazards(self, preds, max_duration, batch_size, verbose, baseline_hazards_,
                                    eval_=True, num_workers=0):
        max_duration = np.inf if max_duration is None else max_duration
#         if baseline_hazards_ is self.baseline_hazards_:
#             bch = self.baseline_cumulative_hazards_
#         else:
        bch = self.compute_baseline_cumulative_hazards(set_hazards=False, 
                                                       baseline_hazards_=baseline_hazards_)
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(preds).reshape(1, -1)
        return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg), 
                            index=bch.index)
    
    
    def compute_baseline_cumulative_hazards(self, preds=None, target=None, max_duration=None, sample=None,
                                            batch_size=8224, set_hazards=True, baseline_hazards_=None,
                                            eval_=True, num_workers=0):
        """See `compute_baseline_hazards. This is the cumulative version."""
        if ((preds is not None) or (target is not None)) and (baseline_hazards_ is not None):
            raise ValueError("'preds', 'target' and 'baseline_hazards_' can not both be different from 'None'.")
        if baseline_hazards_ is None:
            baseline_hazards_ = self.compute_baseline_hazards(preds, target, max_duration, sample, batch_size,
                                                             set_hazards=False, eval_=eval_, num_workers=num_workers)
        assert baseline_hazards_.index.is_monotonic_increasing,\
            'Need index of baseline_hazards_ to be monotonic increasing, as it represents time.'
        bch = (baseline_hazards_
                .cumsum()
                .rename('baseline_cumulative_hazards'))
        if set_hazards:
            self.baseline_hazards_ = baseline_hazards_
            self.baseline_cumulative_hazards_ = bch
        return bch
    
    
    def compute_baseline_hazards(self, preds, df_target, max_duration, batch_size, eval_=True, num_workers=0):
        if max_duration is None:
            max_duration = np.inf

        # Here we are computing when expg when there are no events.
        #   Could be made faster, by only computing when there are events.
        return (df_target
                .assign(expg=np.exp(preds))
                .groupby(self.duration_col)
                .agg({'expg': 'sum', self.event_col: 'sum'})
                .sort_index(ascending=False)
                .assign(expg=lambda x: x['expg'].cumsum())
                .pipe(lambda x: x[self.event_col]/x['expg'])
                .fillna(0.)
                .iloc[::-1]
                .loc[lambda x: x.index <= max_duration]
                .rename('baseline_hazards'))
                
                
                