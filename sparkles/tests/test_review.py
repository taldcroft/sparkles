import numpy as np
import pickle
from pathlib import Path

from proseco import get_aca_catalog
from ..core import ACAReviewTable

KWARGS_48464 = {'att': [-0.51759295, -0.30129397, 0.27093045, 0.75360213],
                'date': '2019:031:13:25:30.000',
                'detector': 'ACIS-S',
                'dither_acq': (7.9992, 7.9992),
                'dither_guide': (7.9992, 7.9992),
                'man_angle': 67.859,
                'n_acq': 8,
                'n_fid': 0,
                'n_guide': 8,
                'obsid': 48464,
                'sim_offset': -3520.0,
                'focus_offset': 0,
                't_ccd_acq': -9.943,
                't_ccd_guide': -9.938}


def test_review_catalog(tmpdir):
    aca = get_aca_catalog(**KWARGS_48464)
    acar = aca.get_review_table()
    acar.run_aca_review()
    assert acar.messages == [
        {'text': 'Guide star imposter offset 2.6, limit 2.5 arcsec', 'category': 'warning',
         'idx': 2},
        {'text': 'P2: 2.84 less than 3.0 for ER', 'category': 'critical'},
        {'text': 'ER count of 9th (8.9 for -9.9C) mag guide stars 1.91 < 3.0',
         'category': 'critical'}]

    assert acar.roll_options is None

    msgs = (acar.messages >= 'critical')
    assert msgs == [
        {'text': 'P2: 2.84 less than 3.0 for ER', 'category': 'critical'},
        {'text': 'ER count of 9th (8.9 for -9.9C) mag guide stars 1.91 < 3.0',
         'category': 'critical'}]

    assert acar.review_status() == -1

    # Check doing a full review for this obsid
    acar.run_aca_review(make_html=True, report_dir=tmpdir, report_level='critical',
                        roll_level='critical')

    path = Path(str(tmpdir))
    assert (path / 'index.html').exists()
    obspath = path / 'obs48464'
    assert (obspath / 'acq' / 'index.html').exists()
    assert (obspath / 'guide' / 'index.html').exists()
    assert (obspath / 'rolls' / 'index.html').exists()


def test_review_roll_options(tmpdir):
    """
    Test that the 'aca' key in the roll_option dict is an ACAReviewTable
    and that the first one has the same messages as the base (original roll)
    version

    :param tmpdir: temp dir supplied by pytest
    :return: None
    """
    # This is a catalog that has a critical message and one roll option
    kwargs = {'att': (160.9272490316051, 14.851572261604668, 99.996111473617802),
              'date': '2019:046:07:16:58.449',
              'detector': 'ACIS-S',
              'dither_acq': (7.9992, 7.9992),
              'dither_guide': (7.9992, 7.9992),
              'focus_offset': 0.0,
              'man_angle': 1.792525648258372,
              'n_acq': 8,
              'n_fid': 3,
              'n_guide': 5,
              'obsid': 21477,
              'sim_offset': 0.0,
              't_ccd_acq': -11.14616454993262,
              't_ccd_guide': -11.150381856818923}

    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(make_html=True, report_dir=tmpdir, roll_level='critical')

    assert len(acar.roll_options) == 2

    # First roll_option is at the same attitude (and roll) as original.  The check
    # code is run again independently but the outcome should be the same.
    assert acar.roll_options[0]['aca'].messages == acar.messages

    for opt in acar.roll_options:
        assert isinstance(opt['aca'], ACAReviewTable)


def test_probs_weak_reference():
    """
    Test issues related to the weak reference to self.acqs within the AcqProbs
    objects in cand_acqs.

    See comment in ACAReviewTable.__init__() for details.

    """
    aca = get_aca_catalog(**KWARGS_48464)

    aca2 = pickle.loads(pickle.dumps(aca))
    assert aca2.acqs is not aca.acqs

    # These fail.  TODO: fix!
    # aca2 = aca.__class__(aca)  # default is copy=True
    # aca2 = deepcopy(aca)

    acar = ACAReviewTable(aca)

    assert aca.guides is not acar.guides
    assert aca.acqs is not acar.acqs


def test_roll_options_with_include_ids():
    """
    Test case from James that was breaking code due to a roll option that puts
    a force_include star outside the FOV.

    """
    kwargs = {'obsid': 48397.0,
              'att': [0.43437703, -0.47822201, -0.68470554, 0.33734053],
              'date': '2019:053:04:05:33.004', 'detector': 'ACIS-S',
              'dither_acq': (7.9992, 2.0016), 'dither_guide': (7.9992, 2.0016),
              'man_angle': 131.2011858838081, 'n_acq': 8, 'n_fid': 0, 'n_guide': 8,
              'sim_offset': 0.0, 'focus_offset': 0.0, 't_ccd_acq': -12.157792574498563,
              't_ccd_guide': -12.17,
              'include_ids_acq': np.array(  # Also tests passing float ids for include
                  [8.13042280e+08, 8.13040960e+08, 8.13044168e+08, 8.12911064e+08,
                   8.12920176e+08, 8.12913936e+08, 8.13043216e+08, 8.13045352e+08]),
              'include_halfws_acq': np.array(
                  [160., 160., 160., 160., 160., 160., 120., 60.])}

    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review(roll_level='all')
    assert len(acar.roll_options) > 1
