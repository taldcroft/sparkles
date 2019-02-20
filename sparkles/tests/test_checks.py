# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import agasc
import numpy as np

from chandra_aca.transform import mag_to_count_rate
from proseco import get_aca_catalog
from proseco.core import StarsTable
from proseco.characteristics import CCD
from proseco.tests.test_common import DARK40, STD_INFO, mod_std_info
from ..core import ACAReviewTable


def test_check_P2():
    """Test the check of acq P2"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=10.25)
    aca = get_aca_catalog(**STD_INFO, stars=stars, dark=DARK40)
    aca = ACAReviewTable(aca)

    # Check P2 for an OR (default obsid=0)
    aca.check_acq_p2()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'less than 2.0 for OR' in msg['text']

    # Check P2 constructed for an ER with stars intended to have P2 > 2 and < 3
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=9.5)
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8, obsid=50000), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_acq_p2()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'less than 3.0 for ER' in msg['text']


def test_guide_count_er():
    """Test the check that an ER has enough fractional guide stars by guide_count"""

    # This configuration should have not enough bright stars
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=9.5)
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8, obsid=50000),
                          stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'ER count of 9th' in msg['text']

    # This configuration should have not enough stars overall
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=[8.5, 8.5, 8.5])
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8, obsid=50000),
                          stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'ER count of guide stars 3.00 < 6.0' in msg['text']

    # And this configuration should have about the bare minumum (of course better
    # to do this with programmatic instead of fixed checks... TODO)
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=6, mag=[8.5, 8.5, 8.5, 10.0, 10.0, 10.0])
    aca = get_aca_catalog(**mod_std_info(obsid=50000, n_fid=0, n_guide=8), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 0

    # This configuration should not warn with too many really bright stars
    # (allowed to have 3 stars brighter than 6.1)
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=8.0)
    stars.add_fake_star(yang=100, zang=100, mag=6.0)
    stars.add_fake_star(yang=1000, zang=-1000, mag=6.0)
    stars.add_fake_star(yang=-2000, zang=-2000, mag=6.0)
    aca = get_aca_catalog(**mod_std_info(obsid=50000, n_fid=0, n_guide=8), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 0

    # This configuration should warn with too many bright stars
    # (has > 3.0 stars brighter than 6.1
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=4, mag=6.0)
    stars.add_fake_star(yang=1000, zang=1000, mag=8.0)
    stars.add_fake_star(yang=-1000, zang=1000, mag=8.0)
    aca = get_aca_catalog(**mod_std_info(obsid=50000, n_fid=0, n_guide=8), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'caution'
    assert 'ER with more than 3 stars brighter than 6.1.' in msg['text']


def test_guide_count_or():
    """Test the check that an OR has enough fractional guide stars by guide_count"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=[7.0, 7.0, 10.3, 10.3, 10.3])
    aca = get_aca_catalog(**mod_std_info(n_fid=3, n_guide=5, obsid=1), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'OR count of guide stars 2.00 < 4.0' in msg['text']

    # This configuration should not warn with too many really bright stars
    # (allowed to have 1 stars brighter than 6.1)
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=8.0)
    stars.add_fake_star(yang=100, zang=100, mag=6.0)
    aca = get_aca_catalog(**mod_std_info(n_fid=3, n_guide=5, obsid=1), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 0

    # This configuration should warn with too many bright stars
    # (has > 1.0 stars brighter than 6.1
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=3, mag=8.0)
    stars.add_fake_star(yang=1000, zang=1000, mag=6.0)
    stars.add_fake_star(yang=-1000, zang=1000, mag=6.0)
    aca = get_aca_catalog(**mod_std_info(n_fid=3, n_guide=5, obsid=1), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_guide_count()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'caution'
    assert 'OR with more than 1 stars brighter than 6.1.' in msg['text']


def test_pos_err_on_guide():
    """Test the check that no guide star has large POS_ERR"""
    stars = StarsTable.empty()
    stars.add_fake_star(id=100, yang=100, zang=-200, POS_ERR=2010, mag=8.0)
    stars.add_fake_star(id=101, yang=0, zang=500, mag=8.0, POS_ERR=1260)  # Just over warning
    stars.add_fake_star(id=102, yang=-200, zang=500, mag=8.0, POS_ERR=1240)  # Just under warning
    stars.add_fake_star(id=103, yang=500, zang=500, mag=8.0, POS_ERR=1260)  # Not selected

    aca = get_aca_catalog(**mod_std_info(n_fid=0), stars=stars, dark=DARK40, raise_exc=True,
                          include_ids_guide=[100, 101])  # Must force 100, 101, pos_err too big

    aca = ACAReviewTable(aca)

    # 103 not selected because pos_err > 1.25 arcsec
    assert aca.guides['id'].tolist() == [100, 101, 102]

    # Run pos err checks
    for guide in aca.guides:
        aca.check_pos_err_guide(guide)

    assert len(aca.messages) == 2
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'Guide star 100 POS_ERR 2.01' in msg['text']

    msg = aca.messages[1]
    assert msg['category'] == 'warning'
    assert 'Guide star 101 POS_ERR 1.26' in msg['text']


def test_guide_edge_check():
    stars = StarsTable.empty()
    dither = 8
    row_lim = CCD['row_max'] - CCD['row_pad'] - CCD['window_pad'] - dither / 5
    col_lim = -(CCD['col_max'] - CCD['col_pad'] - CCD['window_pad'] - dither / 5)

    # Set positions just below or above CCD['guide_extra_pad'] in row / col
    stars.add_fake_star(id=101, mag=8, row=row_lim - 2.9, col=0)
    stars.add_fake_star(id=102, mag=8, row=row_lim - 3.1, col=0)
    stars.add_fake_star(id=103, mag=8, row=row_lim - 5.1, col=0)
    stars.add_fake_star(id=104, mag=8, row=0, col=col_lim + 2.9)
    stars.add_fake_star(id=105, mag=8, row=0, col=col_lim + 3.1)
    stars.add_fake_star(id=106, mag=8, row=0, col=col_lim + 5.1)

    stars.add_fake_constellation(n_stars=6, mag=8.5)

    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8),
                          stars=stars, dark=DARK40, raise_exc=True,
                          include_ids_guide=np.arange(101, 107))
    acar = ACAReviewTable(aca)
    acar.check_catalog()

    assert acar.messages == [
        {'category': 'critical',
         'idx': 5,
         'text': 'Less than 3.0 pix edge margin row lim 495.4 val 492.5 delta 2.9'},
        {'category': 'info',
         'idx': 6,
         'text': 'Less than 5.0 pix edge margin row lim 495.4 val 492.3 delta 3.1'},
        {'category': 'critical',
         'idx': 7,
         'text': 'Less than 3.0 pix edge margin col lim -502.4 val -499.5 delta 2.9'},
        {'category': 'info',
         'idx': 8,
         'text': 'Less than 5.0 pix edge margin col lim -502.4 val -499.3 delta 3.1'}]


def test_imposters_on_guide():
    """Test the check for imposters by adding one imposter to a fake star"""
    stars = StarsTable.empty()
    # Add two stars because separate P2 tests seem to break with just one star
    stars.add_fake_constellation(n_stars=5, mag=9.5)
    mag = 8.0
    cnt = mag_to_count_rate(mag)
    stars.add_fake_star(id=110, row=100, col=-200, mag=mag)
    dark_with_badpix = DARK40.copy()
    dark_with_badpix.aca[100, -200] = cnt * 0.1
    dark_with_badpix.aca[100, -201] = cnt * 0.1
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8), stars=stars, dark=dark_with_badpix,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_imposters_guide(aca.guides.get_id(110))
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'warning'
    assert 'Guide star imposter offset' in msg['text']


def test_bad_star_set():
    bad_id = 1248994952
    star = agasc.get_star(bad_id)
    ra = star['RA']
    dec = star['DEC']
    aca = get_aca_catalog(**mod_std_info(n_fid=0, att=(ra, dec, 0)), dark=DARK40,
                          include_ids_guide=[bad_id])
    acar = ACAReviewTable(aca)
    acar.check_catalog()
    assert acar.messages == [
        {'text': 'Star 1248994952 is in proseco bad star set', 'category': 'critical', 'idx': 5}]


def test_too_bright_guide_magerr():
    """Test the check for too-bright guide stars within mult*mag_err of 5.8"""
    stars = StarsTable.empty()
    # Add two stars because separate P2 tests seem to break with just one star
    stars.add_fake_star(id=100, yang=100, zang=-200, mag=6.0, mag_err=0.11, MAG_ACA_ERR=10)
    stars.add_fake_star(id=101, yang=0, zang=500, mag=8.0)
    aca = get_aca_catalog(**mod_std_info(n_fid=0), stars=stars, dark=DARK40, raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_too_bright_guide(aca.guides.get_id(100))
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert '2*mag_err of 5.8' in msg['text']


def test_check_fid_spoiler_score():
    """Test checking fid spoiler score"""
    stars = StarsTable.empty()
    # def add_fake_stars_from_fid(self, fid_id=1, offset_y=0, offset_z=0, mag=7.0,
    #                            id=None, detector='ACIS-S', sim_offset=0):
    stars.add_fake_constellation(n_stars=8)

    # Add a red spoiler on top of fids 1-4 and yellow on top of fid 5
    stars.add_fake_stars_from_fid(fid_id=[1, 2, 3, 4], mag=7 + 2)
    stars.add_fake_stars_from_fid(fid_id=[5], mag=7 + 4.5)

    aca = get_aca_catalog(stars=stars, **STD_INFO)

    assert np.all(aca.fids.cand_fids['spoiler_score'] == [4, 4, 4, 4, 1, 0])

    acar = ACAReviewTable(aca)
    acar.check_catalog()
    assert acar.messages == [
        {'text': 'Fid 1 has red spoiler: star 108 with mag 9.0', 'category': 'critical', 'idx': 1},
        {'text': 'Fid 5 has yellow spoiler: star 112 with mag 11.5', 'category': 'warning',
         'idx': 2}]


def test_check_fid_count():
    """Test checking fid count"""
    stars = StarsTable.empty()
    # def add_fake_stars_from_fid(self, fid_id=1, offset_y=0, offset_z=0, mag=7.0,
    #                            id=None, detector='ACIS-S', sim_offset=0):
    stars.add_fake_constellation(n_stars=8)

    aca = get_aca_catalog(stars=stars, **mod_std_info(detector='HRC-S', sim_offset=40000))
    acar = ACAReviewTable(aca)
    acar.check_catalog()

    assert acar.messages == [
        {'text': 'Catalog has 2 fids but 3 are expected', 'category': 'critical'}]
