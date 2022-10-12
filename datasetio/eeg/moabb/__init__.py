from moabb.datasets.bnci import BNCI2014001 as moabbBNCI2014001
from moabb.datasets.bnci import BNCI2015001 as moabbBNCI2015001
from moabb.datasets import Lee2019_MI as moabbLee2019

from .base import PreprocessedDataset, CachableDatase
from .stieger2021 import Stieger2021
from .hinss2021 import Hinss2021

class BNCI2014001(moabbBNCI2014001, CachableDatase):

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = super()._get_single_subject_data(subject=subject)
        map = dict(session_T=1,session_E=2)
        sessions = dict([(map[k],v)  for k, v in sessions.items()])
        return sessions

class BNCI2015001(moabbBNCI2015001, CachableDatase):

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = super()._get_single_subject_data(subject=subject)
        map = dict(session_A=1,session_B=2,session_C=3)
        sessions = dict([(map[k],v)  for k, v in sessions.items()])
        return sessions

class Lee2019(moabbLee2019, CachableDatase):

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = super()._get_single_subject_data(subject=subject)
        map = dict(session_1=1,session_2=2)
        sessions = dict([(map[k],v)  for k, v in sessions.items()])
        return sessions
