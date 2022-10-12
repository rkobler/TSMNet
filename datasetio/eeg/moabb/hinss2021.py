import os
import mne
import numpy as np
import pooch
import logging
import requests
import json
import subprocess
import re
import glob

from scipy.io import loadmat
import moabb.datasets.download as dl

from .base import PreprocessedDataset

log = logging.getLogger(__name__)



def doi_to_url(doi, api_url = lambda x : f"https://doi.org/api/handles/{x}?type=URL"):

    url = None
    headers = {"Content-Type": "application/json"}
    response_data = dl.fs_issue_request("GET", api_url(doi), headers=headers)

    if 'values' in response_data:
        candidates = [ val['data']['value']  for val in response_data['values'] if 'data' in val and isinstance(val['data'], dict) and 'value' in val['data']]
        url = candidates[0] if len(candidates)> 0 else None

    return url



def url_get_json(url : str):

    headers = {"Content-Type": "application/json"}
    response = dl.fs_issue_request("GET", url, headers=headers)
    return response



class Hinss2021(PreprocessedDataset):

    ZENODO_JSON_API_URL = lambda x : f"https://zenodo.org/api/{x}"
    
    TASK_TO_EVENTID = dict(RS='rest', MATBeasy='easy', MATBmed='medium', MATBdiff='difficult')

    def __init__(self, interval = [0, 2], channels = None, srate = None):
        super().__init__(
            subjects=list(range(1, 15+1)),
            sessions_per_subject=2,
            events=dict(easy=1, medium=2, difficult=3, rest=4),
            code="Hinss2021",
            interval=interval,
            paradigm="imagery",
            doi="10.5281/zenodo.4917217",
            channels=channels,
            srate=srate
        )



    def preprocess(self, raw):
        # interpolate channels marked as bad
        if len(raw.info['bads']) > 0:
            raw.interpolate_bads()        
        return super().preprocess(raw)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        key_dest = f"MNE-{self.code:s}-data"
        path = os.path.join(dl.get_dataset_path(self.code, path), key_dest)

        url = doi_to_url(self.doi)
        if url is None:
            raise ValueError("Could not find zenodo id based on dataset DOI!")
        
        zenodoid = url.split('/')[-1]

        metadata = url_get_json(Hinss2021.ZENODO_JSON_API_URL(f"records/{zenodoid}"))

        fnames = []
        for record in metadata['files']:

            fname = record['key']
            fpath = os.path.join(path, fname)
            
            
            # metadata
            # if record['type'] != 'zip' and not os.path.exists(fpath): # subject data
            #     pooch.retrieve(record['links']['self'], record['checksum'], fname, path, downloader=pooch.HTTPDownloader(progressbar=True))
            # subject specific data
            if record['type'] == 'zip' and fname == f"P{subject:02d}.zip":
                if not os.path.exists(fpath):
                    files = pooch.retrieve(record['links']['self'], record['checksum'], fname, path, 
                        processor=pooch.Unzip(),
                        downloader=pooch.HTTPDownloader(progressbar=True))
                
                # load the data
                tasks = list(Hinss2021.TASK_TO_EVENTID.keys())
                taskpattern = '('+ '|'.join(tasks)+')'
                pattern = f'{fpath}.unzip/P{subject:02d}/S?/eeg/alldata_*.set'
                candidates = glob.glob(pattern, recursive=True) 
                fnames += [c for c in candidates if re.search(f'.*{taskpattern}.set', c)]

        return fnames


    def _get_single_subject_data(self, subject):
        fnames = self.data_path(subject)

        subject_data = {}
        for fn in fnames:
            meta = re.search('alldata_sbj(?P<subject>\d\d)_sess(?P<session>\d)_((?P<event>\w+))',
                             os.path.basename(fn))
            sid = int(meta['session'])

            if sid not in range(1,self.n_sessions+1):
                continue

            epochs = mne.io.read_epochs_eeglab(fn, verbose=False)
            assert(len(epochs.event_id) == 1)
            event_id = Hinss2021.TASK_TO_EVENTID[list(epochs.event_id.keys())[0]]
            epochs.event_id = {event_id : self.event_id[event_id]}
            epochs.events[:,2] = epochs.event_id[event_id]

            # covnert to continuous raw object with correct annotations
            continuous_data = np.swapaxes(epochs.get_data(),0,1).reshape((len(epochs.info['chs']),-1))
            raw = mne.io.RawArray(data=continuous_data, info=epochs.info, verbose=False, first_samp=1)
            # XXX use standard electrode layout rather than invidividual positions
            # raw.set_montage(epochs.get_montage())
            raw.set_montage('standard_1005')
            events = epochs.events.copy()
            evt_desc = dict(zip(epochs.event_id.values(),epochs.event_id.keys()))

            annot = mne.annotations_from_events(events, raw.info['sfreq'], event_desc=evt_desc, first_samp=1)

            raw.set_annotations(annot)
            
            if sid in subject_data:
                subject_data[sid][0].append(raw)
            else:
                subject_data[sid] = {0 : raw}
            
            # discard boundary annotations
            keep = [i for i, desc in enumerate(subject_data[sid][0].annotations.description) if desc in self.event_id]
            subject_data[sid][0].set_annotations(subject_data[sid][0].annotations[keep])

        return subject_data
