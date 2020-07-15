"""
"""
import os, io, sys
import re
import logging
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
from easydict import EasyDict as ED

from utils.misc import get_record_list_recursive
from utils.scoring_aux_data import (
    dx_mapping_all, dx_mapping_scored, dx_mapping_unscored,
)


__all__ = [
    "CINC2020",
]


class CINC2020(object):
    """ NOT Finished,

    Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020

    ABOUT CINC2020:
    ---------------
    0. There are 6 difference tranches of training data, listed as follows:
        A. 6,877 recordings from China Physiological Signal Challenge in 2018 (CPSC2018):  https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz 
        B. 3,453 recordings from China 12-Lead ECG Challenge Database (unused data from CPSC2018 and NOT the CPSC2018 test data): https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_2.tar.gz
        C. 74 recordings from the St Petersburg INCART 12-lead Arrhythmia Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_StPetersburg.tar.gz 
        D. 516 recordings from the PTB Diagnostic ECG Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_PTB.tar.gz
        E. 21,837 recordings from the PTB-XL electrocardiography Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_PTB-XL.tar.gz
        F. 10,344 recordings from a Georgia 12-Lead ECG Challenge Database: https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz
    In total, 43,101 labeled recordings of 12-lead ECGs from four countries (China, Germany, Russia, and the USA) across 3 continents have been posted publicly for this Challenge, with approximately the same number hidden for testing, representing the largest public collection of 12-lead ECGs

    1. the A tranche training data comes from CPSC2018, whose folder name is `Training_WFDB`. The B tranche training data are unused training data of CPSC2018, having folder name `Training_2`. For these 2 tranches, ref. the docstring of `database_reader.other_databases.cpsc2018.CPSC2018`
    2. C. D. E. tranches of training data all come from corresponding PhysioNet dataset, whose details can be found in corresponding files:
        C: database_reader.physionet_databases.incartdb.INCARTDB
        D: database_reader.physionet_databases.ptbdb.PTBDB
        E: database_reader.physionet_databases.ptb_xl.PTB_XL
    the C tranche has folder name `Training_StPetersburg`, the D tranche has folder name `Training_PTB`, the F tranche has folder name `WFDB`
    3. the F tranche is entirely new, posted for this Challenge, and represents a unique demographic of the Southeastern United States. It has folder name `Training_E/WFDB`.
    4. only a part of diagnosis (diseases that appear in the labels of the 6 tranches of training data) are used in the scoring function (ref. `dx_mapping_scored_cinc2020`), while others are ignored (ref. `dx_mapping_unscored_cinc2020`). The scored diagnoses were chosen based on prevalence of the diagnoses in the training data, the severity of the diagnoses, and the ability to determine the diagnoses from ECG recordings. The ignored diagnosis can be put in a a 'non-class' group.
    5. the (updated) scoring function has a scoring matrix with nonzero off-diagonal elements. This scoring function reflects the clinical reality that some misdiagnoses are more harmful than others and should be scored accordingly. Moreover, it reflects the fact that confusing some classes is much less harmful than confusing other classes.

    NOTE:
    -----
    1. The datasets have been roughly processed to have a uniform format, hence differ from their original resource (e.g. differe in sampling frequency, sample duration, etc.)
    2. The original datasets might have richer metadata (especially those from PhysioNet), which can be fetched from corresponding reader's docstring or website of the original source
    3. Each sub-dataset might have its own organizing scheme of data, which should be carefully dealt with
    4. There are few 'absolute' diagnoses in 12 lead ECGs, where large discrepancies in the interpretation of the ECG can be found even inspected by experts. There is inevitably something lost in translation, especially when you do not have the context. This doesn't mean making an algorithm isn't important
    5. The labels are noisy, which one has to deal with in all real world data

    ISSUES:
    -------

    Usage:
    ------
    1. ECG arrhythmia detection

    References:
    -----------
    [1] https://physionetchallenges.github.io/2020/
    [2] http://2018.icbeb.org/#
    [3] https://physionet.org/content/incartdb/1.0.0/
    [4] https://physionet.org/content/ptbdb/1.0.0/
    [5] https://physionet.org/content/ptb-xl/1.0.1/
    """
    def __init__(self, db_dir:Optional[str]=None, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str, optional,
            storage path of the database
            if not specified, data will be fetched from Physionet
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        self.db_name = 'CINC2020'
        self.working_dir = working_dir or os.getcwd()
        self.verbose = verbose
        self.logger = None
        self._set_logger(prefix=self.db_name)

        self.db_dir_base = self.db_dir
        self.db_dirs = ED({
            "A": os.path.join(self.db_dir_base, "Training_WFDB"),
            "B": os.path.join(self.db_dir_base, "Training_2"),
            "C": os.path.join(self.db_dir_base, "Training_StPetersburg"),
            "D": os.path.join(self.db_dir_base, "Training_PTB"),
            "E": os.path.join(self.db_dir_base, "WFDB"),
            "F": os.path.join(self.db_dir_base, "Training_E", "WFDB"),
        })
        self.all_records = ED({
            tranche: get_record_list_recursive(self.db_dirs[tranche]) for tranche in "ABCDEF"
        })
        self.rec_prefix = ED({
            "A": "A", "B": "Q", "C": "I", "D": "S", "E": "HR", "F": "E",
        })
        """
        prefixes can be obtained using the following code:
        >>> pfs = ED({k:set() for k in "ABCDEF"})
        >>> for k, p in db_dir.items():
        >>>     af = os.listdir(p)
        >>>     for fn in af:
        >>>         pfs[k].add("".join(re.findall(r"[A-Z]", os.path.splitext(fn)[0])))
        """
        self.rec_ext = '.mat'
        self.ann_ext = '.hea'

        # self.freq = 500
        # self.spacing = 1000 / self.freq

        self.all_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',]

        self.df_ecg_arrhythmia = dx_mapping_all[['Dx','SNOMED CT Code','Abbreviation']]


    def get_subject_id(self, rec:str) -> int:
        """ not finished,

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        sid: int,
            the `get_subject_id` corr. to `rec`
        """
        sid = 0
        raise NotImplementedError


    def _set_logger(self, prefix:Optional[str]=None):
        """

        Parameters:
        -----------
        prefix: str, optional,
            prefix (for each line) of the logger, and its file name
        """
        _prefix = prefix+"-" if prefix else ""
        self.logger = logging.getLogger('{}-{}-logger'.format(_prefix, self.db_name))
        log_filepath = os.path.join(self.working_dir, "{}{}.log".format(_prefix, self.db_name))
        print("log file path is set {}".format(log_filepath))

        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_filepath)
        if self.verbose >= 2:
            print("levels of c_handler and f_handler are set DEBUG")
            c_handler.setLevel(logging.DEBUG)
            f_handler.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        elif self.verbose >= 1:
            print("level of c_handler is set INFO, level of f_handler is set DEBUG")
            c_handler.setLevel(logging.INFO)
            f_handler.setLevel(logging.DEBUG)
            self.logger.setLevel(logging.DEBUG)
        else:
            print("levels of c_handler and f_handler are set WARNING")
            c_handler.setLevel(logging.WARNING)
            f_handler.setLevel(logging.WARNING)
            self.logger.setLevel(logging.WARNING)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)


    def _get_tranche(self, rec:str) -> str:
        """
        """
        prefix = "".join(re.findall(r"[A-Z]", "rec"))
        return {v:k for k,v in self.rec_prefix.items()}[prefix]


    def load_data(self, rec:str, data_format='channels_last') -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        data_format: str, default 'channels_last',
            format of the ecg data, 'channels_last' or 'channels_first' (original)
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        tranche = self._get_tranche(rec)
        rec_fp = os.path.join(self.db_dirs[tranche], rec + self.rec_ext)
        data = loadmat(rec_fp)
        data = np.asarray(data['val'], dtype=np.float64)
        if data_format == 'channels_last':
            data = data.T
        
        return data

    
    def load_ann(self, rec:str) -> dict:
        """ finished, checked,
        
        Parameters:
        -----------
        rec: str,
            name of the record
        
        Returns:
        --------
        ann_dict, dict,
            the annotations with items: ref. self.ann_items
        """
        tranche = self._get_tranche(rec)
        ann_fp = os.path.join(self.db_dirs[tranche], rec + self.ann_ext)
        with open(ann_fp, 'r') as f:
            header_data = f.read().splitlines()

        ann_dict = {}
        ann_dict['rec_name'], ann_dict['nb_leads'], ann_dict['freq'], ann_dict['nb_samples'], ann_dict['datetime'], daytime = header_data[0].split(' ')
        ann_dict['nb_leads'] = int(ann_dict['nb_leads'])
        ann_dict['freq'] = int(ann_dict['freq'])
        ann_dict['nb_samples'] = int(ann_dict['nb_samples'])
        ann_dict['datetime'] = datetime.strptime(' '.join([ann_dict['datetime'], daytime]), '%d-%b-%Y %H:%M:%S')
        try: # see NOTE. 1.
            ann_dict['age'] = int([l for l in header_data if l.startswith('#Age')][0].split(": ")[-1])
        except:
            ann_dict['age'] = np.nan
        ann_dict['sex'] = [l for l in header_data if l.startswith('#Sex')][0].split(": ")[-1]
        ann_dict['diagnosis_Dx'] = [l for l in header_data if l.startswith('#Dx')][0].split(": ")[-1].split(",")
        try:
            ann_dict['diagnosis_Dx'] = [int(item) for item in ann_dict['diagnosis_Dx']]
            selection = dx_mapping_all['SNOMED CT code'].isin(ann_dict['diagnosis_Dx'])
            ann_dict['diagnosis'] = dx_mapping_all[selection]['Abbreviation'].tolist()
            ann_dict['diagnosis_fullname'] = dx_mapping_all[selection]['dx'].tolist()
        except:  # the old version, the Dx's are abbreviations
            ann_dict['diagnosis'] = ann_dict['diagnosis_Dx']
            selection = dx_mapping_all['Abbreviation'].isin(ann_dict['diagnosis'])
            ann_dict['diagnosis_fullname'] = dx_mapping_all[selection]['dx'].tolist()
        # if not keep_original:
        #     for idx, d in enumerate(ann_dict['diagnosis']):
        #         if d in ['Normal', 'SNR']:
        #             ann_dict['diagnosis'] = ['N']
        ann_dict['medical_prescription'] = [l for l in header_data if l.startswith('#Rx')][0].split(": ")[-1]
        ann_dict['history'] = [l for l in header_data if l.startswith('#Hx')][0].split(": ")[-1]
        ann_dict['symptom_or_surgery'] = [l for l in header_data if l.startswith('#Sx')][0].split(": ")[-1]
        df_leads = pd.read_csv(io.StringIO('\n'.join(header_data[1:13])), delim_whitespace=True, header=None)
        df_leads.columns = ['filename', 'res+offset', 'resolution(mV)', 'ADC', 'baseline', 'first_value', 'checksum', 'redundant', 'lead_name']
        df_leads['resolution(bits)'] = df_leads['res+offset'].apply(lambda s: s.split('+')[0])
        df_leads['offset'] = df_leads['res+offset'].apply(lambda s: s.split('+')[1])
        df_leads = df_leads[['filename', 'resolution(bits)', 'offset', 'resolution(mV)', 'ADC', 'baseline', 'first_value', 'checksum', 'lead_name']]
        df_leads['resolution(mV)'] = df_leads['resolution(mV)'].apply(lambda s: s.split('/')[0])
        for k in ['resolution(bits)', 'offset', 'resolution(mV)', 'ADC', 'baseline', 'first_value', 'checksum']:
            df_leads[k] = df_leads[k].apply(lambda s: int(s))
        ann_dict['df_leads'] = df_leads

        return ann_dict

    
    def get_labels(self, rec:str, fullname:bool=False) -> List[str]:
        """ finished, checked,
        
        Parameters:
        -----------
        rec: str,
            name of the record
        
        Returns:
        --------
        labels, list,
            the list of labels (abbr. diagnosis)
        """
        ann_dict = self.load_ann(rec)
        if fullname:
            labels = ann_dict['diagnosis_fullname']
        else:
            labels = ann_dict['diagnosis']
        return labels

    
    def get_patient_info(self, rec:str, items:Optional[List[str]]=None) -> dict:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        items: list of str, optional,
            items of the patient information (e.g. sex, age, etc.)
        
        Returns:
        --------
        patient_info, dict,
        """
        if items is None or len(items) == 0:
            info_items = [
                'age', 'sex', 'medical_prescription', 'history', 'symptom_or_surgery',
            ]
        else:
            info_items = items
        ann_dict = self.load_ann(rec)
        patient_info = [ann_dict[item] for item in info_items]

        return patient_info


    def save_challenge_predictions(self, rec:str, output_dir:str, scores:List[Real], labels:List[int], classes:List[str]) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        output_dir: str,
            directory to save the predictions
        scores: list of real,
            ...
        labels: list of int,
            0 or 1
        classes: list of str,
            ...
        """
        new_file = rec + '.csv'
        output_file = os.path.join(output_dir, new_file)

        # Include the filename as the recording number
        recording_string = '#{}'.format(rec)
        class_string = ','.join(classes)
        label_string = ','.join(str(i) for i in labels)
        score_string = ','.join(str(i) for i in scores)

        with open(output_file, 'w') as f:
            # f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')
            f.write("\n".join([recording_string, class_string, label_string, score_string, ""]))


    def plot(self, rec:str, leads:Optional[Union[str, List[str]]]=None, **kwargs):
        """ not finished, not checked,

        Parameters:
        -----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to 
        kwargs: dict,
        """
        tranche = self._get_tranche(rec)
        if tranche in "CDE":
            physionet_lightwave_suffix = ED({
                "C": "incartdb/1.0.0",
                "D": "ptbdb/1.0.0",
                "E": "ptb-xl/1.0.1",
            })
            url = f"https://physionet.org/lightwave/?db={physionet_lightwave_suffix[tranche]}"
            print("better view: {}\n"*3)
            
        if 'plt' not in dir():
            import matplotlib.pyplot as plt
        if leads is None or leads == 'all':
            leads = self.all_leads
        assert all([l in self.all_leads for l in leads])

        lead_list = self.load_ann(rec)['df_leads']['lead_name'].tolist()
        lead_indices = [lead_list.index(l) for l in leads]
        data = self.load_data(rec)[lead_indices]
        y_ranges = np.max(np.abs(data), axis=1) + 100

        diag = self.get_diagnosis(rec, full_name=False)

        nb_leads = len(leads)

        t = np.arange(data.shape[1]) / self.freq
        duration = len(t) / self.freq
        fig_sz_w = int(round(4.8 * duration))
        fig_sz_h = 6 * y_ranges / 1500
        fig, axes = plt.subplots(nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        for idx in range(nb_leads):
            axes[idx].plot(t, data[idx], label='lead - ' + leads[idx] + '\n' + 'labels - ' + ",".join(diag))
            axes[idx].axhline(y=0, linestyle='-', linewidth='1.0', color='red')
            axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
            axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
            axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
            axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
            axes[idx].grid(which='major', linestyle='-', linewidth='0.5', color='red')
            axes[idx].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            axes[idx].legend(loc='best')
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(-y_ranges[idx], y_ranges[idx])
            axes[idx].set_xlabel('Time [s]')
            axes[idx].set_ylabel('Voltage [μV]')
        plt.subplots_adjust(hspace=0.2)
        plt.show()
