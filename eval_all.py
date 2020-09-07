"""
"""
from itertools import repeat
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from cfg import ModelCfg
from data_reader import CINC2020Reader as CR
from dataset import CINC2020
from utils.scoring_metrics import evaluate_12ECG_score
from driver import load_challenge_data
from run_12ECG_classifier import run_12ECG_classifier


__all__ = [
    "eval_all",
    "eval_all_parallel",
]


def eval_all(tranches:Optional[str]=None) -> pd.DataFrame:
    """
    """
    models = load_12ECG_model()
    dr = CR(TrainCfg.db_dir)
    ds_config = deepcopy(TrainCfg)
    if tranches:
        ds_config.tranches_for_training = tranches
    ds = CINC2020(config=ds_config, training=False)

    truth_labels, truth_array = [], []
    binary_predictions, scalar_predictions = [], []
    classes = ModelCfg.full_classes
    with tqdm(ds.records, total=len(ds.records)) as t:
        for rec in t:
            data_fp = dr.get_data_filepath(rec)
            data, header_data = load_challenge_data(data_fp)
            current_label, current_score, _ = \
                run_12ECG_classifier(data, header_data, hehe_models, verbose=0)
            binary_predictions.append(current_label)
            scalar_predictions.append(current_score)
            tl = dr.get_labels(rec, fmt='a')
            ta = list(repeat(0, len(classes)))
            for c in tl:
                ta[classes.index(c)] = 1
            truth_labels.append(tl)
            truth_array.append(ta)
    
    # gather results into a DataFrame
    print("gathering results into a `DataFrame`...")
    df_eval_res = pd.DataFrame(scalar_predictions)
    df_eval_res.columns = classes
    df_eval_res['binary_predictions'] = ''
    df_eval_res['truth_labels'] = ''
    classes = np.array(classes)
    for idx, row in df_eval_res.iterrows():
        df_eval_res.at[idx, 'binary_predictions'] = \
            classes[np.where(binary_predictions[idx]==1)[0]].tolist()
        df_eval_res.at[idx, 'truth_labels'] = truth_labels[idx]
    class = classes.tolist()

    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric = \
        evaluate_12ECG_score(
            classes=classes,
            truth=truth_array,
            scalar_pred=scalar_predictions,
            binary_pred=binary_predictions,
        )
    msg = f"""
        results on tranches {tranches or 'all'}:
        ------------------------------
        auroc:              {auroc}
        auprc:              {auprc}
        accuracy:           {accuracy}
        f_measure:          {f_measure}
        f_beta_measure:     {f_beta_measure}
        g_beta_measure:     {g_beta_measure}
        challenge_metric:   {challenge_metric}
        ----------------------------------------
    """
    print(msg)  # in case no logger

    return df_eval_res


def eval_all_parallel(tranches:Optional[str]=None) -> pd.DataFrame:
    """
    """
    raise NotImplementedError
