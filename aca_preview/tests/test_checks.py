# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from proseco import get_aca_catalog
from proseco.core import StarsTable
from proseco.tests.test_common import DARK40, STD_INFO
from aca_preview.preview import ACAReviewTable


def test_check_P2():
    """Test the check of acq P2"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=10.25)
    aca = get_aca_catalog(**STD_INFO, stars=stars, dark=DARK40)
    ACAReviewTable.add_review_methods(aca)

    aca.check_acq_p2()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'less than 2.0 for OR' in msg['text']
