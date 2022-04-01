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

raw_cache = getCache('part2ch11_raw')

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'age_months, series_uid',
)


@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    nifti_list = glob.glob('/home/feczk001/shared/projects/S1067_Loes/experiments/scans/*.nii.gz')
    presentOnDisk_set = {os.path.split(p)[-1][:-7] for p in nifti_list}

    candidateInfo_list = []
    with open('/home/feczk001/shared/projects/S1067_Loes/experiments/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            age_in_months = row[1]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            candidateInfo_list.append(CandidateInfoTuple(
                age_in_months,
                series_uid,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


class Mri:
    def __init__(self, series_uid):
        mri_path = f'/home/feczk001/shared/projects/S1067_Loes/experiments/scans/{series_uid}.nii.gz'
        img = nib.load(mri_path)
        self.struct_arr = np.array(img.get_data(), dtype=np.float32)


@functools.lru_cache(1, typed=True)
def getMri(series_uid):
    return Mri(series_uid)


def getMriRawCandidate(series_uid):
    mri_path = f'/home/feczk001/shared/projects/S1067_Loes/experiments/scans/{series_uid}.nii.gz'
    img = nib.load(mri_path)
    candidate_a = np.array(img.get_data(), dtype=np.float32)

    return candidate_a
