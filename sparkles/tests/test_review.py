from pathlib import Path

from proseco import get_aca_catalog


def test_review_catalog(tmpdir):
    kwargs = {'att': [-0.51759295, -0.30129397, 0.27093045, 0.75360213],
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

    aca = get_aca_catalog(**kwargs)
    acar = aca.get_review_table()
    acar.run_aca_review()
    assert acar.messages == [
        {'category': 'warning',
         'idx': 2,
         'text': 'Guide star imposter offset 2.6, limit 2.5 arcsec'},
        {'category': 'critical',
         'idx': 7,
         'text': 'Less than 2.5 pix edge margin row lim -495.4 val -495.0 delta 0.4'},
        {'category': 'warning',
         'idx': 7,
         'text': 'Guide star imposter offset 2.6, limit 2.5 arcsec'},
        {'category': 'critical', 'text': 'P2: 2.84 less than 3.0 for ER'},
        {'category': 'critical',
         'text': 'ER bright stars: only 2 stars brighter than 9.0'},
        {'category': 'critical', 'text': 'ER guide stars: only 7 stars'}]

    assert acar.roll_options is None

    msgs = (acar.messages >= 'critical')
    assert msgs == [
        {'category': 'critical',
         'idx': 7,
         'text': 'Less than 2.5 pix edge margin row lim -495.4 val -495.0 delta 0.4'},
        {'category': 'critical', 'text': 'P2: 2.84 less than 3.0 for ER'},
        {'category': 'critical',
         'text': 'ER bright stars: only 2 stars brighter than 9.0'},
        {'category': 'critical', 'text': 'ER guide stars: only 7 stars'}]

    assert acar.review_status() == -1

    # Check doing a full review for this obsid
    acar.run_aca_review(report_dir=tmpdir, roll_level='critical', report_level='critical')

    path = Path(str(tmpdir))
    assert (path / 'index.html').exists()
    obspath = path / 'obs48464'
    assert (obspath / 'acq' / 'index.html').exists()
    assert (obspath / 'guide' / 'index.html').exists()
    assert (obspath / 'rolls' / 'index.html').exists()
