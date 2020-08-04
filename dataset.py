"""
"""
import os, io, sys
import re
import json
import time
import logging
# import pprint
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
from easydict import EasyDict as ED

import utils
from utils.misc import (
    get_record_list_recursive,
    dict_to_str,
    ms2samples,
)
from utils.scoring_aux_data import (
    dx_mapping_all, dx_mapping_scored, dx_mapping_unscored,
    normalize_class,
)
from utils import ecg_arrhythmia_knowledge as EAK
from cfg import PlotCfg


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
    4. only a part of diagnosis_abbr (diseases that appear in the labels of the 6 tranches of training data) are used in the scoring function (ref. `dx_mapping_scored_cinc2020`), while others are ignored (ref. `dx_mapping_unscored_cinc2020`). The scored diagnoses were chosen based on prevalence of the diagnoses in the training data, the severity of the diagnoses, and the ability to determine the diagnoses from ECG recordings. The ignored diagnosis_abbr can be put in a a 'non-class' group.
    5. the (updated) scoring function has a scoring matrix with nonzero off-diagonal elements. This scoring function reflects the clinical reality that some misdiagnoses are more harmful than others and should be scored accordingly. Moreover, it reflects the fact that confusing some classes is much less harmful than confusing other classes.

    6. sampling frequencies:
        A. (CPSC2018): 500 Hz
        B. (CPSC2018-2): 500 Hz
        C. (INCART): 257 Hz
        D. (PTB): 1000 Hz
        E. (PTB-XL): 500 Hz
        F. (Georgia): 500 Hz
    7. all data are recorded in the leads ordering of
        ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    using for example the following code:
    >>> db_dir = "/media/cfs/wenhao71/data/cinc2020_data/"
    >>> working_dir = "./working_dir"
    >>> data_gen = CINC2020(db_dir=db_dir,working_dir=working_dir)
    >>> set_leads = []
    >>> for tranche, l_rec in data_gen.all_records.items():
    ...     for rec in l_rec:
    ...         ann = data_gen.load_ann(rec)
    ...         leads = ann['df_leads']['lead_name'].values.tolist()
    ...     if leads not in set_leads:
    ...         set_leads.append(leads)

    NOTE:
    -----
    1. The datasets have been roughly processed to have a uniform format, hence differ from their original resource (e.g. differe in sampling frequency, sample duration, etc.)
    2. The original datasets might have richer metadata (especially those from PhysioNet), which can be fetched from corresponding reader's docstring or website of the original source
    3. Each sub-dataset might have its own organizing scheme of data, which should be carefully dealt with
    4. There are few 'absolute' diagnoses in 12 lead ECGs, where large discrepancies in the interpretation of the ECG can be found even inspected by experts. There is inevitably something lost in translation, especially when you do not have the context. This doesn't mean making an algorithm isn't important
    5. The labels are noisy, which one has to deal with in all real world data
    6. each line of the following classes are considered the same (in the scoring matrix):
        - RBBB, CRBBB (NOT including IRBBB)
        - PAC, SVPB
        - PVC, VPB
    7. unfortunately, the newly added tranches (C - F) have baseline drift and are much noisier. In contrast, CPSC data have had baseline removed and have higher SNR
    8. on Aug. 1, 2020, adc gain (including 'resolution', 'ADC'? in .hea files) of datasets INCART, PTB, and PTB-xl (tranches C, D, E) are corrected. After correction, (the .tar files of) the 3 datasets are all put in a "WFDB" subfolder. In order to keep the structures consistant, they are moved into "Training_StPetersburg", "Training_PTB", "WFDB" as previously. Using the following code, one can check the resolutions and baselines of each tranche:
    >>> db_dir = "/media/cfs/wenhao71/data/cinc2020_data/"
    >>> working_dir = "./working_dir"
    >>> data_gen = CINC2020(db_dir=db_dir,working_dir=working_dir)
    >>> resolution = {tranche: set() for tranche in "ABCDEF"}
    >>> baseline = {tranche: set() for tranche in "ABCDEF"}
    >>> for tranche, l_rec in data_gen.all_records.items():
    ...     for rec in l_rec:
    ...         ann = data_gen.load_ann(rec)
    ...         resolution[tranche] = resolution[tranche].union(set(ann['df_leads']['resolution(mV)']))
    ...         baseline[tranche] = baseline[tranche].union(set(ann['df_leads']['baseline']))
    >>> print(resolution, baseline)
    {'A': {1000}, 'B': {1000}, 'C': {1000}, 'D': {1000}, 'E': {1000}, 'F': {4880}} {'A': {0}, 'B': {0}, 'C': {0}, 'D': {0}, 'E': {0}, 'F': {0}}
    9. the .mat files all contain digital signals, which has to be converted to physical values using adc gain, basesline, etc. in corresponding .hea files. `wfdb.rdrecord` has already done this conversion, hence greatly simplifies the data loading process.
    NOTE that there's a difference when using `wfdb.rdrecord`: data from `loadmat` are in 'channel_first' format, while `wfdb.rdrecord.p_signal` produces data in the 'channel_last' format

    ISSUES:
    -------
    1. reading the .hea files, baselines of all records are 0, however it is not the case if one plot the signal
    2. about half of the LAD records satisfy the '2-lead' criteria, but fail for the '3-lead' criteria, which means that their axis is (-30°, 0°) which is not truely LAD

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
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        Parameters:
        -----------
        db_dir: str,
            storage path of the database
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        self.db_name = 'CINC2020'
        self.working_dir = working_dir or os.getcwd()
        os.makedirs(self.working_dir, exist_ok=True)
        self.verbose = verbose
        self.logger = None
        self._set_logger(prefix=self.db_name)

        self.rec_ext = 'mat'
        self.ann_ext = 'hea'

        self.db_dir_base = db_dir
        self.db_dirs = ED({
            "A": os.path.join(self.db_dir_base, "Training_WFDB"),
            "B": os.path.join(self.db_dir_base, "Training_2"),
            "C": os.path.join(self.db_dir_base, "Training_StPetersburg"),
            "D": os.path.join(self.db_dir_base, "Training_PTB"),
            "E": os.path.join(self.db_dir_base, "WFDB"),
            "F": os.path.join(self.db_dir_base, "Training_E", "WFDB"),
        })

        self._all_records = None
        self._ls_rec()
        self._diagnoses_records_list = None
        self._ls_diagnoses_records()

        self.rec_prefix = ED({
            "A": "A", "B": "Q", "C": "I", "D": "S", "E": "HR", "F": "E",
        })
        """
        prefixes can be obtained using the following code:
        >>> pfs = ED({k:set() for k in "ABCDEF"})
        >>> for k, p in db_dir.items():
        ...     af = os.listdir(p)
        ...     for fn in af:
        ...         pfs[k].add("".join(re.findall(r"[A-Z]", os.path.splitext(fn)[0])))
        """
        self.tranche_names = ED({
            "A": "CPSC",
            "B": "CPSC-Extra",
            "C": "StPetersburg",
            "D": "PTB",
            "E": "PTB-XL",
            "F": "Georgia",
        })
        self.freq = {
            "A": 500, "B": 500, "C": 257, "D": 1000, "E": 500, "F": 500,
        }
        self.spacing = {t: 1000 / f for t,f in self.freq.items()}

        self.all_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',]

        self.df_ecg_arrhythmia = dx_mapping_all[['Dx','SNOMED CT Code','Abbreviation']]
        self.ann_items = [
            'rec_name', 'nb_leads','freq','nb_samples','datetime','age','sex',
            'diagnosis','df_leads',
            'medical_prescription','history','symptom_or_surgery',
        ]


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
        s2d = {"A":"11", "B":"12", "C":"21", "D":"31", "E":"32", "F":"41"}
        s2d = {self.rec_prefix[k]:v for k,v in s2d.items()}
        prefix = "".join(re.findall(r"[A-Z]", rec))
        n = rec.replace(prefix,"")
        sid = int(f"{s2d[prefix]}{'0'*(8-len(n))}{n}")
        return sid

    
    def _ls_rec(self) -> NoReturn:
        """ finished, checked,

        list all the records and load into `self._all_records`,
        facilitating further uses
        """
        fn = "record_list.json"
        record_list_fp = os.path.join(self.db_dir_base, fn)
        if not os.path.isfile(record_list_fp):
            record_list_fp = os.path.join(utils._BASE_DIR, "utils", fn)
        if os.path.isfile(record_list_fp):
            with open(record_list_fp, "r") as f:
                self._all_records = json.load(f)
        else:
            print("Please wait patiently to let the reader find all records of all the tranches...")
            start = time.time()
            self._all_records = ED({
                tranche: get_record_list_recursive(self.db_dirs[tranche], self.rec_ext) \
                    for tranche in "ABCDEF"
            })
            print(f"Done in {time.time() - start} seconds!")
            with open(record_list_fp, "w") as f:
                json.dump(self._all_records, f)


    @property
    def all_records(self):
        """ finished, checked
        """
        if self._all_records is None:
            self._ls_rec()
        return self._all_records


    def _ls_diagnoses_records(self) -> NoReturn:
        """ finished, checked,

        list all the records for all diagnoses
        """
        fn = "diagnoses_records_list.json"
        dr_fp = os.path.join(self.db_dir_base, fn)
        if not os.path.isfile(dr_fp):
            dr_fp = os.path.join(utils._BASE_DIR, "utils", fn)
        if os.path.isfile(dr_fp):
            with open(dr_fp, "r") as f:
                self._diagnoses_records_list = json.load(f)
        else:
            print("Please wait patiently to let the reader list records for each diagnosis...")
            start = time.time()
            self._diagnoses_records_list = {d: [] for d in df_weights_abbr.columns.values.tolist()}
            for tranche, l_rec in self.all_records.items():
                for rec in l_rec:
                    ann = self.load_ann(rec)
                    ld = ann["diagnosis_scored"]['diagnosis_abbr']
                    for d in ld:
                        self._diagnoses_records_list[d].append(rec)
            print(f"Done in {time.time() - start} seconds!")


    @property
    def diagnoses_records_list(self):
        """ finished, checked
        """
        if self._diagnoses_records_list is None:
            self._ls_diagnoses_records()
        return self._diagnoses_records_list


    def _set_logger(self, prefix:Optional[str]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        prefix: str, optional,
            prefix (for each line) of the logger, and its file name
        """
        _prefix = prefix+"-" if prefix else ""
        self.logger = logging.getLogger(f'{_prefix}-{self.db_name}-logger')
        log_filepath = os.path.join(self.working_dir, f"{_prefix}{self.db_name}.log")
        print(f"log file path is set {log_filepath}")

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
        """ finished, checked,

        get the tranche's symbol (one of 'A','B','C','D','E','F') of a record via its name

        Parameters:
        -----------
        rec: str,
            name of the record

        Returns:
        --------
        tranche, str,
            symbol of the tranche, ref. `self.rec_prefix`
        """
        prefix = "".join(re.findall(r"[A-Z]", rec))
        tranche = {v:k for k,v in self.rec_prefix.items()}[prefix]
        return tranche


    def load_data(self, rec:str, leads:Optional[Union[str, List[str]]]=None, data_format='channel_first', backend:str='wfdb', units:str='mV') -> np.ndarray:
        """ finished, checked,

        load physical (converted from digital) ecg data,
        which is more understandable for humans

        Parameters:
        -----------
        rec: str,
            name of the record
        leads: str or list of str, optional,
            the leads to load
        data_format: str, default 'channel_first',
            format of the ecg data,
            'channel_last' (alias 'lead_last'), or
            'channel_first' (alias 'lead_first', original)
        backend: str, default 'wfdb',
            the backend data reader, can also be 'scipy'
        units: str, default 'mV',
            units of the output signal, can also be 'μV', with an alias of 'uV'
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        assert data_format.lower() in ['channel_first', 'lead_first', 'channel_last', 'lead_last']
        tranche = self._get_tranche(rec)
        if backend.lower() == 'scipy':
            rec_fp = os.path.join(self.db_dirs[tranche], f'{rec}.{self.rec_ext}')
            data = loadmat(rec_fp)
            resolutions = self.load_ann(rec, raw=False)['df_leads']['resolution(mV)'].values.reshape(12,-1)
            data = np.asarray(data['val'], dtype=np.float64) / resolutions
        elif backend.lower() == 'wfdb':
            rec_fp = os.path.join(self.db_dirs[tranche], rec)
            data = np.asarray(wfdb.rdrecord(rec_fp).p_signal.T, dtype=np.float64)

        if leads and isinstance(leads, str):
            leads_ind = [self.all_leads.index(leads)]
            data = data[leads_ind,:]
        elif leads:
            leads_ind = [self.all_leads.index(item) for item in leads]
            data = data[leads_ind,:]

        if data_format.lower() in ['channel_last', 'lead_last']:
            data = data.T

        if units.lower() in ['uv', 'μv']:
            data = data * 1000
        
        return data

    
    def load_ann(self, rec:str, raw:bool=False) -> Union[dict,str]:
        """ finished, checked,

        load annotations (header) stored in the .hea files
        
        Parameters:
        -----------
        rec: str,
            name of the record
        raw: bool, default False,
            if True, the raw annotations without parsing will be returned
        
        Returns:
        --------
        ann_dict, dict or str,
            the annotations with items: ref. `self.ann_items`
        """
        tranche = self._get_tranche(rec)
        ann_fp = os.path.join(self.db_dirs[tranche], f'{rec}.{self.ann_ext}')
        with open(ann_fp, 'r') as f:
            header_data = f.read().splitlines()
        
        if raw:
            ann_dict = '\n'.join(header_data)
            return ann_dict

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

        ann_dict['diagnosis'] = dict()
        ann_dict['diagnosis_scored'] = dict()
        ann_dict['diagnosis']['diagnosis_code'] = [l for l in header_data if l.startswith('#Dx')][0].split(": ")[-1].split(",")
        try:
            ann_dict['diagnosis']['diagnosis_code'] = [int(item) for item in ann_dict['diagnosis']['diagnosis_code']]
            # selection = dx_mapping_all['SNOMED CT Code'].isin(ann_dict['diagnosis']['diagnosis_code'])
            # ann_dict['diagnosis']['diagnosis_abbr'] = dx_mapping_all[selection]['Abbreviation'].tolist()
            # ann_dict['diagnosis']['diagnosis_fullname'] = dx_mapping_all[selection]['Dx'].tolist()
            ann_dict['diagnosis']['diagnosis_abbr'] = \
                [ dx_mapping_all[dx_mapping_all['SNOMED CT Code']==dc]['Abbreviation'].values[0] \
                    for dc in ann_dict['diagnosis']['diagnosis_code'] ]
            ann_dict['diagnosis']['diagnosis_fullname'] = \
                [ dx_mapping_all[dx_mapping_all['SNOMED CT Code']==dc]['Dx'].values[0] \
                    for dc in ann_dict['diagnosis']['diagnosis_code'] ]
            scored_indices = np.isin(ann_dict['diagnosis']['diagnosis_code'], dx_mapping_scored['SNOMED CT Code'].values)
            ann_dict['diagnosis_scored']['diagnosis_code'] = \
                [ item for idx, item in enumerate(ann_dict['diagnosis']['diagnosis_code']) \
                    if scored_indices[idx] ]
            ann_dict['diagnosis_scored']['diagnosis_abbr'] = \
                [ item for idx, item in enumerate(ann_dict['diagnosis']['diagnosis_abbr']) \
                    if scored_indices[idx] ]
            ann_dict['diagnosis_scored']['diagnosis_fullname'] = \
                [ item for idx, item in enumerate(ann_dict['diagnosis']['diagnosis_fullname']) \
                    if scored_indices[idx] ]
        except:  # the old version, the Dx's are abbreviations
            ann_dict['diagnosis']['diagnosis_abbr'] = ann_dict['diagnosis']['diagnosis_code']
            selection = dx_mapping_all['Abbreviation'].isin(ann_dict['diagnosis']['diagnosis_abbr'])
            ann_dict['diagnosis']['diagnosis_fullname'] = dx_mapping_all[selection]['Dx'].tolist()
        # if not keep_original:
        #     for idx, d in enumerate(ann_dict['diagnosis_abbr']):
        #         if d in ['Normal', 'SNR']:
        #             ann_dict['diagnosis_abbr'] = ['N']

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
        df_leads.index = df_leads['lead_name']
        df_leads.index.name = None
        ann_dict['df_leads'] = df_leads

        return ann_dict


    def load_header(self, rec:str, raw:bool=False) -> Union[dict,str]:
        """
        alias for `load_ann`, as annotations are also stored in header files
        """
        return self.load_ann(rec, raw)

    
    def get_labels(self, rec:str, scored_only:bool=True, abbr:bool=True) -> List[str]:
        """ finished, checked,

        read labels (diagnoses or arrhythmias) of a record
        
        Parameters:
        -----------
        rec: str,
            name of the record
        scored_only: bool, default True,
            only get the labels that are scored in the CINC2020 official phase
        abbr: bool, default True,
            labels in abbreviations or fullnames
        
        Returns:
        --------
        labels, list,
            the list of labels
        """
        ann_dict = self.load_ann(rec)
        if scored_only:
            labels = ann_dict['diagnosis_scored']
        else:
            labels = ann_dict['diagnosis']
        if abbr:
            labels = labels['diagnosis_abbr']
        else:
            labels = labels['diagnosis_fullname']
        return labels


    def get_freq(self, rec:str) -> Real:
        """ finished, checked,

        get the sampling frequency of a record

        Paramters:
        ----------
        rec: str,
            name of the record

        Returns:
        --------
        freq: real number,
            sampling frequency of the record `rec`
        """
        tranche = self._get_tranche(rec)
        freq = self.freq[tranche]
        return freq

    
    def get_subject_info(self, rec:str, items:Optional[List[str]]=None) -> dict:
        """ finished, checked,

        read auxiliary information of a subject (a record) stored in the header files

        Parameters:
        -----------
        rec: str,
            name of the record
        items: list of str, optional,
            items of the subject's information (e.g. sex, age, etc.)
        
        Returns:
        --------
        subject_info: dict,
            information about the subject, including
            'age', 'sex', 'medical_prescription', 'history', 'symptom_or_surgery',
        """
        if items is None or len(items) == 0:
            info_items = [
                'age', 'sex', 'medical_prescription', 'history', 'symptom_or_surgery',
            ]
        else:
            info_items = items
        ann_dict = self.load_ann(rec)
        subject_info = [ann_dict[item] for item in info_items]

        return subject_info


    def save_challenge_predictions(self, rec:str, output_dir:str, scores:List[Real], labels:List[int], classes:List[str]) -> NoReturn:
        """ finished, checked, 
        
        TODO: update for the official phase

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
        new_file = f'{rec}.csv'
        output_file = os.path.join(output_dir, new_file)

        # Include the filename as the recording number
        recording_string = f'#{rec}'
        class_string = ','.join(classes)
        label_string = ','.join(str(i) for i in labels)
        score_string = ','.join(str(i) for i in scores)

        with open(output_file, 'w') as f:
            # f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')
            f.write("\n".join([recording_string, class_string, label_string, score_string, ""]))


    def plot(self, rec:str, data:Optional[np.ndarray]=None, ticks_granularity:int=0, leads:Optional[Union[str, List[str]]]=None, same_range:bool=False, waves:Optional[Dict[str, Sequence[int]]]=None, **kwargs) -> NoReturn:
        """ finished, checked, to improve,

        plot the signals of a record or external signals (units in μV),
        with metadata (freq, labels, tranche, etc.),
        possibly also along with wave delineations

        Parameters:
        -----------
        rec: str,
            name of the record
        data: ndarray, optional,
            12-lead ecg signal to plot,
            if given, data of `rec` will not be used
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more
        leads: str or list of str, optional,
            the leads to plot
        same_range: bool, default False,
            if True, forces all leads to have the same y range
        waves: dict, optional,
            indices of the wave critical points, including
            'p_onsets', 'p_peaks', 'p_offsets',
            'q_onsets', 'q_peaks', 'r_peaks', 's_peaks', 's_offsets',
            't_onsets', 't_peaks', 't_offsets'
        kwargs: dict,

        TODO:
        -----
        1. slice too long records, and plot separately for each segment
        2. plot waves using `axvspan`

        NOTE:
        -----
        `Locator` of `plt` has default `MAXTICKS` equal to 1000,
        if not modifying this number, at most 40 seconds of signal could be plotted once

        Contributors: Jeethan, and WEN Hao
        """
        tranche = self._get_tranche(rec)
        if tranche in "CDE":
            physionet_lightwave_suffix = ED({
                "C": "incartdb/1.0.0",
                "D": "ptbdb/1.0.0",
                "E": "ptb-xl/1.0.1",
            })
            url = f"https://physionet.org/lightwave/?db={physionet_lightwave_suffix[tranche]}"
            print(f"better view: {url}")
            
        if 'plt' not in dir():
            import matplotlib.pyplot as plt
            plt.MultipleLocator.MAXTICKS = 3000
        if leads is None or leads == 'all':
            leads = self.all_leads
        assert all([l in self.all_leads for l in leads])

        # lead_list = self.load_ann(rec)['df_leads']['lead_name'].tolist()
        # lead_indices = [lead_list.index(l) for l in leads]
        lead_indices = [self.all_leads.index(l) for l in leads]
        if data is None:
            _data = self.load_data(rec, data_format='channel_first', units='μV')[lead_indices]
        else:
            units = self._auto_detect_units(data)
            print(f"input data is auto detected to have units in {units}")
            if units.lower() == 'mv':
                _data = 1000 * data
            else:
                _data = data
        
        if same_range:
            y_ranges = np.ones((_data.shape[0],)) * np.max(np.abs(_data)) + 100
        else:
            y_ranges = np.max(np.abs(_data), axis=1) + 100

        if waves:
            if waves.get('p_onsets', None) and waves.get('p_offsets', None):
                p_waves = [
                    [onset, offset] for onset, offset in zip(waves['p_onsets'], waves['p_offsets'])
                ]
            elif waves.get('p_peaks', None):
                p_waves = [
                    [max(0, p + ms2samples(PlotCfg.p_onset)), min(_data.shape[1], p + ms2samples(PlotCfg.p_offset))] \
                        for p in waves['p_peaks']
                ]
            else:
                p_waves = []
            if waves.get('q_onsets', None) and waves.get('s_offsets', None):
                qrs = [
                    [onset, offset] for onset, offset in zip(waves['q_onsets'], waves['s_offsets'])
                ]
            elif waves.get('q_peaks', None) and waves.get('s_peaks', None):
                qrs = [
                    [max(0, q + ms2samples(PlotCfg.q_onset)), min(_data.shape[1], s + ms2samples(PlotCfg.s_offset))] \
                        for q,s in zip(waves['q_peaks'], waves['s_peaks'])
                ]
            elif waves.get('r_peaks', None):
                qrs = [
                    [max(0, r + ms2samples(PlotCfg.qrs_radius)), min(_data.shape[1], r + ms2samples(PlotCfg.qrs_radius))] \
                        for r in waves['r_peaks']
                ]
            else:
                qrs = []
            if waves.get('t_onsets', None) and waves.get('t_offsets', None):
                t_waves = [
                    [onset, offset] for onset, offset in zip(waves['t_onsets'], waves['t_offsets'])
                ]
            elif waves.get('t_peaks', None):
                t_waves = [
                    [max(0, t + ms2samples(PlotCfg.t_onset)), min(_data.shape[1], t + ms2samples(PlotCfg.t_offset))] \
                        for t in waves['t_peaks']
                ]
            else:
                t_waves = []
        else:
            p_waves, qrs, t_waves = [], [], []
        palette = {'p_waves': 'green', 'qrs': 'red', 't_waves': 'pink',}
        plot_alpha = 0.4

        diag_scored = self.get_labels(rec, scored_only=True, abbr=True)
        diag_all = self.get_labels(rec, scored_only=False, abbr=True)

        nb_leads = len(leads)

        seg_len = self.freq[tranche] * 25  # 25 seconds
        nb_segs = _data.shape[1] // seg_len

        t = np.arange(_data.shape[1]) / self.freq[tranche]
        duration = len(t) / self.freq[tranche]
        fig_sz_w = int(round(4.8 * duration))
        fig_sz_h = 6 * y_ranges / 1500
        nl = "\n"
        fig, axes = plt.subplots(nb_leads, 1, sharex=True, figsize=(fig_sz_w, np.sum(fig_sz_h)))
        for idx in range(nb_leads):
            # axes[idx].plot(t, _data[idx], label='lead - ' + leads[idx] + '\n' + 'labels - ' + ",".join(diag_scored))
            # axes[idx].plot(t, _data[idx], label=f'lead - {leads[idx]}{nl}labels_s - {",".join(diag_scored)}{nl}labels_a - {",".join(diag_all)}')
            axes[idx].plot(t, _data[idx], label=f'lead - {leads[idx]}')
            axes[idx].axhline(y=0, linestyle='-', linewidth='1.0', color='red')
            # NOTE that `Locator` has default `MAXTICKS` equal to 1000
            if ticks_granularity >= 1:
                axes[idx].xaxis.set_major_locator(plt.MultipleLocator(0.2))
                axes[idx].yaxis.set_major_locator(plt.MultipleLocator(500))
                axes[idx].grid(which='major', linestyle='-', linewidth='0.5', color='red')
            if ticks_granularity >= 2:
                axes[idx].xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                axes[idx].yaxis.set_minor_locator(plt.MultipleLocator(100))
                axes[idx].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            # add extra info. to legend
            # https://stackoverflow.com/questions/16826711/is-it-possible-to-add-a-string-as-a-legend-item-in-matplotlib
            axes[idx].plot([], [], ' ', label=f"labels_s - {','.join(diag_scored)}")
            axes[idx].plot([], [], ' ', label=f"labels_a - {','.join(diag_all)}")
            axes[idx].plot([], [], ' ', label=f"tranche - {self.tranche_names[tranche]}")
            axes[idx].plot([], [], ' ', label=f"freq - {self.freq[tranche]}")
            for w in ['p_waves', 'qrs', 't_waves']:
                for itv in eval(w):
                    axes[idx].axvspan(itv[0], itv[1], color=palette[w], alpha=plot_alpha)
            axes[idx].legend(loc='upper left')
            axes[idx].set_xlim(t[0], t[-1])
            axes[idx].set_ylim(-y_ranges[idx], y_ranges[idx])
            axes[idx].set_xlabel('Time [s]')
            axes[idx].set_ylabel('Voltage [μV]')
        plt.subplots_adjust(hspace=0.2)
        plt.show()


    def _auto_detect_units(self, data:np.ndarray) -> str:
        """ finished, checked

        automatically detect units of `data`,
        under the assumption that `data` not raw data, with baseline removed

        Parameters:
        -----------
        data: ndarray,
            the data to detect its units

        Returns:
        --------
        units: str,
            units of `data`, 'μV' or 'mV'
        """
        _MAX_mV = 10  # 10mV
        max_val = np.max(np.abs(data))
        if max_val > _MAX_mV:
            units = 'μV'
        else:
            units = 'mV'
        return units


    @classmethod
    def get_arrhythmia_knowledge(cls, arrhythmias:Union[str,List[str]], **kwargs) -> NoReturn:
        """ finished, checked,

        knowledge about ECG features of specific arrhythmias,

        Parameters:
        -----------
        arrhythmias: str, or list of str,
            the arrhythmia(s) to check, in abbreviations or in SNOMED CT Code
        """
        if isinstance(arrhythmias, str):
            d = [normalize_class(arrhythmias)]
        else:
            d = [normalize_class(c) for c in arrhythmias]
        # pp = pprint.PrettyPrinter(indent=4)
        # unsupported = [item for item in d if item not in dx_mapping_all['Abbreviation']]
        unsupported = [item for item in d if item not in dx_mapping_scored['Abbreviation'].values]
        assert len(unsupported) == 0, \
            f"{unsupported} {'is' if len(unsupported)==1 else 'are'} not supported!"
        for idx, item in enumerate(d):
            # pp.pprint(eval(f"EAK.{item}"))
            print(dict_to_str(eval(f"EAK.{item}")))
            if idx < len(d)-1:
                print("*"*110)
