import re


from ctfeval.config import params
from ctfeval.datasets import (
    get_lemon_subject_ids,
    get_rift_subject_ids,
    rift_subfolder,
    get_lemon_age,
)


def test_get_lemon_subject_ids():
    """
    Check the expected format sub-XXXXXX
    """
    subject_ids = get_lemon_subject_ids(params.lemon_bids)
    match_format = [re.fullmatch("sub-\\d{6}", sid) for sid in subject_ids]
    assert all(match_format), "Expected subject IDs to match sub-XXXXXX regex"
    assert "sub-010229" not in subject_ids, "sub-010229 should be excluded"
    assert "sub-032437" not in subject_ids, "sub-032437 should be excluded"


def test_get_lemon_age():
    assert get_lemon_age("20-25") == 22.5
    assert get_lemon_age("60-65") == 62.5


def test_get_rift_subject_ids():
    subject_ids = get_rift_subject_ids()

    assert len(subject_ids) == 12
    assert all("sub" in subject_id for subject_id in subject_ids)

    subject_ids = get_rift_subject_ids(strip_sub=True)

    assert len(subject_ids) == 12
    assert all(isinstance(subject_id, int) for subject_id in subject_ids)


def test_rift_subfolder():
    assert rift_subfolder(1, 1) == "tag_type_1_random_phases_1"
