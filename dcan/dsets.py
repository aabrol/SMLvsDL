import csv
import functools
import glob
import logging
import os
from collections import namedtuple

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'age_months, id',
)


@functools.lru_cache(1)
def get_candidate_info_list(require_on_disk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    nifti_list = glob.glob('/home/feczk001/shared/projects/S1067_Loes/experiments/scans/*.nii.gz')
    present_on_disk_set = {os.path.split(p)[-1][:-7] for p in nifti_list}

    candidate_info_list = []
    with open('/home/feczk001/shared/projects/S1067_Loes/experiments/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            ident = row[0]
            age_in_months = row[1]

            if ident not in present_on_disk_set and require_on_disk_bool:
                continue

            candidate_info_list.append(CandidateInfoTuple(
                age_in_months,
                ident,
            ))

    candidate_info_list.sort()
    return candidate_info_list
