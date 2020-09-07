"""
"""
import time
from itertools import repeat
from copy import deepcopy
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing as mp

from cfg import ModelCfg, TrainCfg
from data_reader import CINC2020Reader as CR
from dataset import CINC2020
from models.utils.torch_utils import default_collate_fn as collate_fn
from utils.scoring_metrics import evaluate_12ECG_score
from driver import load_challenge_data
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier


__all__ = [
    "eval_all",
    "eval_all_parallel",
]


if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)
    _DTYPE = torch.float64
else:
    _DTYPE = torch.float32


def eval_all(tranches:Optional[str]=None) -> pd.DataFrame:
    """ finished, checked,

    Parameters:
    -----------
    tranches: str, optional,
        tranches for making the evaluation,
        can be one of "AB", "E", "F", or None (None defaults to "ABEF")
    """
    models = load_12ECG_model()
    dr = CR(TrainCfg.db_dir)
    ds_config = deepcopy(TrainCfg)
    if tranches:
        ds_config.tranches_for_training = tranches
    ds = CINC2020(config=ds_config, training=False)

    time.sleep(3)

    truth_labels, truth_array = [], []
    binary_predictions, scalar_predictions = [], []
    classes = ModelCfg.full_classes
    # ds.records = ds.records[:10]
    with tqdm(ds.records, total=len(ds.records)) as t:
        for rec in t:
            data_fp = dr.get_data_filepath(rec)
            data, header_data = load_challenge_data(data_fp)
            current_label, current_score, _ = \
                run_12ECG_classifier(data, header_data, models, verbose=0)
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
    classes = classes.tolist()

    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric = \
        evaluate_12ECG_score(
            classes=classes,
            truth=np.array(truth_array),
            scalar_pred=np.array(scalar_predictions),
            binary_pred=np.array(binary_predictions),
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
    """ NOT finished, not checked,
    """
    batch_size = 16

    loaded_models = load_12ECG_model()
    dr = CR(TrainCfg.db_dir)
    ds_config = deepcopy(TrainCfg)
    if tranches:
        ds_config.tranches_for_training = tranches
    ds = CINC2020(config=ds_config, training=False)
    data_loader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    truth_array = np.array([]).reshape(0, len(ModelCfg.full_classes))
    binary_predictions = np.array([]).reshape(0, len(ModelCfg.full_classes))
    scalar_predictions = np.array([]).reshape(0, len(ModelCfg.full_classes))

    with tqdm(total=n_train) as pbar:
        for step, (signals, labels) in enumerate(data_loader):
            signals = signals.to(device=device, dtype=_DTYPE)
            labels = labels.numpy()

            dl_scores = []
            for subset, model in loaded_models.items():
                model.eval()
                subset_scores, subset_bin = model.inference(signals)
                if subset in ModelCfg.tranche_classes.keys():
                    subset_scores = extend_predictions(
                        subset_scores,
                        ModelCfg.tranche_classes[subset],
                        ModelCfg.dl_classes,
                    )
                subset_scores = subset_scores[0]  # remove the batch dimension
                dl_scores.append(subset_scores)

            if "NSR" in ModelCfg.dl_classes:
                dl_nsr_cid = ModelCfg.dl_classes.index("NSR")
            elif "426783006" in ModelCfg.dl_classes:
                dl_nsr_cid = ModelCfg.dl_classes.index("426783006")
            else:
                dl_nsr_cid = None

            # TODO: make a classifier using the scores from the 4 different dl models
            dl_scores = np.max(np.array(dl_scores), axis=0)
            dl_conclusions = (dl_scores >= ModelCfg.bin_pred_thr).astype(int)

            # treat exceptional cases
            max_prob = dl_scores.max()
            if max_prob < ModelCfg.bin_pred_nsr_thr and dl_nsr_cid is not None:
                dl_conclusions[row_idx, dl_nsr_cid] = 1
            elif dl_conclusions.sum() == 0:
                dl_conclusions = ((dl_scores+ModelCfg.bin_pred_look_again_tol) >= max_prob)
                dl_conclusions = (dl_conclusions & (dl_scores >= ModelCfg.bin_pred_nsr_thr))
                dl_conclusions = dl_conclusions.astype(int)

            dl_scores = extend_predictions(
                dl_scores,
                ModelCfg.dl_classes,
                ModelCfg.full_classes,
            )
            dl_conclusions = extend_predictions(
                dl_conclusions,
                ModelCfg.dl_classes,
                ModelCfg.full_classes,
            )

            with mp.Pool(processes=batch_size) as pool:
                partial_conclusion = pool.starmap(
                    func=_run_special_detector_once,
                    iterable=signals.tolist(),
                )

            pbar.update(signals.shape[0])


def _run_special_detector_once(data:Sequence):
    """ for running `eval_all_parallel`
    """
    partial_conclusion = special_detectors(np.array(data), ModelCfg.fs, sig_fmt="lead_first")
    is_brady = partial_conclusion.is_brady
    is_tachy = partial_conclusion.is_tachy
    is_LAD = partial_conclusion.is_LAD
    is_RAD = partial_conclusion.is_RAD
    is_PR = partial_conclusion.is_PR
    is_LQRSV = partial_conclusion.is_LQRSV

    tmp = np.zeros(shape=(len(ModelCfg.full_classes,)))
    tmp[ModelCfg.full_classes.index('Brady')] = int(is_brady)
    tmp[ModelCfg.full_classes.index('LAD')] = int(is_LAD)
    tmp[ModelCfg.full_classes.index('RAD')] = int(is_RAD)
    tmp[ModelCfg.full_classes.index('PR')] = int(is_PR)
    tmp[ModelCfg.full_classes.index('LQRSV')] = int(is_LQRSV)
    partial_conclusion = tmp

    return partial_conclusion
