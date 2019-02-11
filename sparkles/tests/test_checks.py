# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from chandra_aca.transform import mag_to_count_rate
from proseco import get_aca_catalog
from proseco.core import StarsTable
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
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.obsid = 50000
    aca.check_acq_p2()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'less than 3.0 for ER' in msg['text']


def test_enough_guide():
    """Test the check that an ER has enough guide stars"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=5, mag=8.0)
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.obsid = 50000
    aca.check_enough_guide_for_ers()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'ER guide stars: only' in msg['text']


def test_bright_guide():
    """Test the check that an ER has enough bright guide stars"""
    stars = StarsTable.empty()
    stars.add_fake_constellation(n_stars=8, mag=9.5)
    aca = get_aca_catalog(**mod_std_info(obsid=50000, n_fid=0, n_guide=8), stars=stars, dark=DARK40,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_bright_guide_for_ers()
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'ER bright stars: only 0 stars brighter than 9.0' in msg['text']


def test_pos_err_on_guide():
    """Test the check that no guide star has large POS_ERR"""
    stars = StarsTable.empty()
    # Add two stars because separate P2 tests seem to break with just one star
    stars.add_fake_star(id=100, yang=100, zang=-200, POS_ERR=2500, mag=8.0)
    stars.add_fake_star(id=101, yang=0, zang=500, mag=8.0)
    aca = get_aca_catalog(**mod_std_info(n_fid=0), stars=stars, dark=DARK40, raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_pos_err_guide(aca.guides.get_id(100))
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'Guide star 100 POS_ERR 2.50' in msg['text']


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
    dark_with_badpix.aca[101, -201] = cnt * 0.1
    dark_with_badpix.aca[101, -200] = cnt * 0.1
    aca = get_aca_catalog(**mod_std_info(n_fid=0, n_guide=8), stars=stars, dark=dark_with_badpix,
                          raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_imposters_guide(aca.guides.get_id(110))
    assert len(aca.messages) == 1
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'Guide star imposter offset' in msg['text']


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


def test_too_bright_guide_mag_aca_err():
    """Test the check for too-bright guide stars with small MAG_ACA_ERR"""
    stars = StarsTable.empty()
    # Add two stars because separate P2 tests seem to break with just one star
    stars.add_fake_star(id=100, yang=100, zang=-200, mag=6.0, mag_err=0.02, MAG_ACA_ERR=0)
    stars.add_fake_star(id=101, yang=0, zang=500, mag=8.0)
    aca = get_aca_catalog(**mod_std_info(n_fid=0), stars=stars, dark=DARK40, raise_exc=True)
    aca = ACAReviewTable(aca)
    aca.check_too_bright_guide(aca.guides.get_id(100))
    msg = aca.messages[0]
    assert msg['category'] == 'critical'
    assert 'small MAG_ACA_ERR' in msg['text']
