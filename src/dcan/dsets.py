import copy
import csv
import functools
import glob
import os
import random
from collections import namedtuple

import nibabel as nib
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('dcan_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'series_uid, eid_str, ses_str, age_int, sex_str, t1_t2_str, run_x_try_int, smriPath_str, motionQCscore_int, passfail_str, notes_str'
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mri_list = glob.glob('data-unversioned/abcd/*.nii.gz')
    presentOnDisk_set = {os.path.split(p)[-1][:-7] for p in mri_list}

    candidateInfo_list = []
    with open('data/ABCD/qc_with_paths.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            eid_ses_t1_t2_run_x_try_uid = '_'.join([row[0], row[1], 'run-' + row[5].zfill(3), row[4].upper() + 'w'])

            if eid_ses_t1_t2_run_x_try_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            eid_str = row[0]
            ses_str = row[1]
            age_int = int(row[2])
            sex_str = row[3]
            t1_t2_str = row[4]
            run_x_try_int = int(row[5])
            smriPath_str = row[6]
            motionQCscore_int = int(row[7])
            passfail_str = row[8]
            notes_str = row[9]

            candidateInfo_list.append(CandidateInfoTuple(
                eid_ses_t1_t2_run_x_try_uid,
                eid_str,
                ses_str,
                age_int,
                sex_str,
                t1_t2_str,
                run_x_try_int,
                smriPath_str,
                motionQCscore_int,
                passfail_str,
                notes_str,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Mri:
    def __init__(self, smri_path):
        mri_path = glob.glob(
            'data-unversioned/abcd/{}.nii.gz'.format(smri_path)
        )[0]

        mri_nii_gz = nib.load(mri_path)
        mri_a = np.array(mri_nii_gz.get_fdata(), dtype=np.float32)

        self.smri_path = smri_path
        self.hu_a = mri_a

    def getRawCandidate(self):
        mri_chunk = self.hu_a

        return mri_chunk


@functools.lru_cache(1, typed=True)
def getMri(series_uid):
    return Mri(series_uid)


@raw_cache.memoize(typed=True)
def getMriRawCandidate(series_uid):
    mri = getMri(series_uid)
    mri_chunk = mri.getRawCandidate()
    return mri_chunk


class ABCDDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
            ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup = self.candidateInfo_list[ndx]

        candidate_a = getMriRawCandidate(
            candidateInfo_tup.series_uid,
        )
        candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        age_t = torch.tensor([
                candidateInfo_tup.age_int
            ],
            dtype=torch.int
        )

        return candidate_t, age_t, candidateInfo_tup.series_uid
