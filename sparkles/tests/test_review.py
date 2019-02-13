import gc
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
        {'category': 'warning',
         'idx': 2,
         'text': 'Guide star imposter offset 2.6, limit 2.5 arcsec'},
        {'category': 'critical', 'text': 'P2: 2.84 less than 3.0 for ER'},
        {'category': 'critical', 'text': 'ER count of 9th mag guide stars 1.57 < 3.0'}]

    assert acar.roll_options is None

    msgs = (acar.messages >= 'critical')
    assert msgs == [
        {'category': 'critical', 'text': 'P2: 2.84 less than 3.0 for ER'},
        {'category': 'critical', 'text': 'ER count of 9th mag guide stars 1.57 < 3.0'}]

    assert acar.review_status() == -1

    # Check doing a full review for this obsid
    acar.run_aca_review(report_dir=tmpdir, roll_level='critical', report_level='critical')

    path = Path(str(tmpdir))
    assert (path / 'index.html').exists()
    obspath = path / 'obs48464'
    assert (obspath / 'acq' / 'index.html').exists()
    assert (obspath / 'guide' / 'index.html').exists()
    assert (obspath / 'rolls' / 'index.html').exists()


def test_probs_weak_reference():
    """
    Test issues related to the weak reference to self.acqs within the AcqProbs
    objects in cand_acqs.

    See comment in ACAReviewTable.__init__() for details.

    """
    aca = get_aca_catalog(**KWARGS_48464)

    aca2 = pickle.loads(pickle.dumps(aca))
    assert aca2.acqs is not aca.acqs
    for probs in aca2.acqs.cand_acqs['probs']:
        assert probs.acqs() is aca2.acqs

    # These fail.  TODO: fix!
    # aca2 = aca.__class__(aca)  # default is copy=True
    # aca2 = deepcopy(aca)

    acar = ACAReviewTable(aca)

    assert aca.guides is not acar.guides
    assert aca.acqs is not acar.acqs

    del aca
    gc.collect()

    for probs in acar.acqs.cand_acqs['probs']:
        assert probs.acqs() is acar.acqs
