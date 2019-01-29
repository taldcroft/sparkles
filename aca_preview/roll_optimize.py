# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Roll optimization during preliminary review of ACA catalogs.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from astropy.table import Table, vstack
from chandra_aca.star_probs import acq_success_prob
from chandra_aca.transform import (radec_to_yagzag, yagzag_to_pixels,
                                   calc_aca_from_targ, calc_targ_from_aca,
                                   snr_mag_for_t_ccd)
from Quaternion import Quat

import proseco.characteristics as ACA
from proseco import get_aca_catalog


ROLL_TABLE = Table.read(str(Path(__file__).parent / 'pitch_rolldev.csv'), format='ascii.basic',
                        guess=False, delimiter=',')


# Grab from chandra_aca PR #71
def guide_count(mags, t_ccd, is_ER=False):
    """Calculate a guide star fractional count/metric using signal-to-noise scaled
    mag thresholds.
    This uses a modification of the guide star fractional counts that were
    suggested at the 7-Mar-2018 SSAWG and agreed upon at the 21-Mar-2018
    SSAWG.  The implementation here does a piecewise linear interpolation
    between the reference mag - fractional count points instead of the
    original "threshold interpolation" (nearest neighbor mag <= reference
    mag).  Approved at 16-Jan-2019 SSAWG.
    One feature is the slight incline in the guide_count curve from 1.0005 at
    mag=6.0 to 1.0 at mag=10.0.  This does not show up in standard outputs
    of guide_counts to two decimal places (8 * 0.0005 = 0.004), but helps with
    minimization.
    :returns: fractional count
    """
    # Generate interpolation curve for the specified input ``t_ccd``
    ref_t_ccd = -10.9
    ref_mags0 = np.array([10.0, 10.2, 10.3, 10.4]) - (1.2 if is_ER else 0.0)
    ref_mags_t_ccd = [snr_mag_for_t_ccd(t_ccd, ref_mag, ref_t_ccd) for ref_mag in ref_mags0]

    # The 5.85 and 5.95 limits are not temperature dependent, these reflect the
    # possibility that the star will be brighter than 5.8 mag and the OBC will
    # reject it.  Note that around 6th mag mean observed catalog error is
    # around 0.1 mag.
    ref_mags = ([5.85, 5.95] + ref_mags_t_ccd)
    ref_counts = [0.0, 1.0005, 1.0, 0.75, 0.5, 0.0]

    # Do the interpolation, noting that np.interp will use the end ``counts``
    # values for any ``mag`` < ref_mags[0] or > ref_mags[-1].
    count = np.sum(np.interp(mags, ref_mags, ref_counts))

    return count


def allowed_rolldev(pitch):
    """Get allowed roll deviation (off-nominal roll) for the given ``pitch``.
    This uses the OFLS table and is an approximation to the true planning limit.
    This is basically the same as https://github.com/sot/Ska.Sun/pull/5.
    :param pitch: Sun pitch angle (deg)
    :returns: Roll deviation (deg)
    """
    idx = np.searchsorted(ROLL_TABLE['pitch'], pitch, side='right')
    return ROLL_TABLE['rolldev'][idx - 1]


def logical_intervals(vals, x=None):
    """
    Determine contiguous intervals during which ``vals`` is True.
    Returns an Astropy Table with a row for each interval.  Columns are:
    * idx_start: index of interval start
    * idx_stop: index of interval stop
    * x_start: x value at idx_start (if ``x`` is supplied)
    * x_stop: x value at idx_stop (if ``x`` is supplied)
    :param vals: bool values for which intervals are returned.
    :param x: x-axis values corresponding to ``vals``
    :returns: Table of intervals
    """
    if len(vals) < 2:
        raise ValueError('Filtered data length must be at least 2')

    transitions = np.concatenate([[True], vals[:-1] != vals[1:], [True]])

    state_vals = vals[transitions[1:]]
    state_idxs = np.where(transitions)[0]

    intervals = {'idx_start': state_idxs[:-1],
                 'idx_stop': state_idxs[1:] - 1}

    out = Table(intervals, names=sorted(intervals))

    # Filter only the True states
    out = out[state_vals]

    if x is not None:
        out['x_start'] = x[out['idx_start']]
        out['x_stop'] = x[out['idx_stop']]

    return out


class RollOptimizeMixin:

    def get_candidate_better_stars(self):
        """Find stars that *might* substantially improve guide or acq catalogs.
        Get stars that might be candidates at a different roll.  This takes
        stars outside the original square CCD FOV (but made smaller by 40
        pixels) and inside a circle corresponding to the box corners (but made
        bigger by 40 pixels).  The inward padding ensures any stars that were
        originally excluded because of dither size etc are considered.
        :returns: list of indexes into self.stars
        """
        # First define a spatial mask ``sp_ok`` on ``stars`` that is the
        # region (mentioned above) between an inner square and outer circle.
        rc_pad = 40
        stars = self.stars
        in_fov = ((np.abs(stars['row']) < ACA.CCD['row_max'] - rc_pad) &
                  (np.abs(stars['col']) < ACA.CCD['col_max'] - rc_pad))
        radius2 = stars['row'] ** 2 + stars['col'] ** 2
        sp_ok = ~in_fov & (radius2 < 2 * (512 + rc_pad) ** 2)

        # Find potential acq stars that are noticably better than worst acq
        # star via the p_acq_model metric, defined as p_acq is at least 0.3
        # better.
        acq_ok = ((stars['CLASS'] == 0) &
                  (stars['mag'] > 5.9) &
                  (stars['mag'] < 11.0) &
                  (~np.isclose(stars['COLOR1'], 0.7)) &
                  (stars['mag_err'] < 1.0) &  # Mag err < 1.0 mag
                  (stars['ASPQ1'] < 40) &  # Less than 2 arcsec offset from nearby spoiler
                  (stars['ASPQ2'] == 0) &  # Proper motion less than 0.5 arcsec/yr
                  (stars['POS_ERR'] < 3000) &  # Position error < 3.0 arcsec
                  ((stars['VAR'] == -9999) | (stars['VAR'] == 5)))  # Not known to vary > 0.2 mag
        idxs = np.flatnonzero(sp_ok & acq_ok)
        p_acqs = acq_success_prob(date=self.acqs.date, t_ccd=self.acqs.t_ccd,
                                  mag=stars['mag'][idxs], color=stars['COLOR1'][idxs],
                                  spoiler=False, halfwidth=120)
        worst_p_acq = min(acq['probs'].p_acq_model(120) for acq in self.acqs)
        ok = p_acqs > worst_p_acq + 0.3
        better_acq_idxs = idxs[ok]

        # Find potential guide stars that are noticably better than worst guide
        # star, defined as being at least 0.2 mag brighter.
        guide_ok = ((stars['CLASS'] == 0) &
                    (stars['mag'] > 5.9) &
                    (stars['mag'] < 10.3) &
                    (stars['mag_err'] < 1.0) &  # Mag err < 1.0 mag
                    (stars['ASPQ1'] < 20) &  # Less than 1 arcsec offset from nearby spoiler
                    (stars['ASPQ2'] == 0) &  # Proper motion less than 0.5 arcsec/yr
                    (stars['POS_ERR'] < 3000) &  # Position error < 3.0 arcsec
                    ((stars['VAR'] == -9999) | (stars['VAR'] == 5)))  # Not known to vary > 0.2 mag

        idxs = np.flatnonzero(sp_ok & guide_ok)
        worst_mag = np.max(self.guides['mag'])
        ok = stars['mag'][idxs] < worst_mag - 0.2
        better_guide_idxs = idxs[ok]

        # Take the union of better stars and return the indexes
        return sorted(set(better_guide_idxs) | set(better_acq_idxs))

    def get_better_rolls(self, cand_idxs, roll_nom=None, roll_dev=None,
                         y_off=0, z_off=0, d_roll=0.25):
        """Find a list of rolls that might substantially improve guide or acq catalogs.
        If ``roll_nom`` is not specified then an approximate value is computed
        via Ska.Sun for the catalog ``date``.  if ``roll_dev`` (max allowed
        off-nominal roll) is not specified it is computed using the OFLS table.
        These will not precisely match ORviewer results.

        :param roll_nom: nominal roll for observation (deg)
        :param roll_dev: max allowed deviation from nominal roll (deg)
        :param y_off: Y offset (deg, sign per OR-list convention)
        :param z_off: Z offset (deg, sign per OR-list convention)
        :param d_roll: step size for examining roll range (deg, default=0.25)

        :returns: list of candidate rolls

        """
        cols = ['id', 'ra', 'dec']
        acqs = Table(self.acqs[cols])
        acqs.meta.clear()

        # Mask for guide star IDs that are also in acqs
        overlap = np.in1d(self.guides['id'], acqs['id'])
        guides = Table(self.guides[cols][~overlap])
        guides.meta.clear()
        cands = vstack([acqs, guides, self.stars[cols][cand_idxs]])

        q_att = Quat(self.att)

        def get_ids_list(roll_offsets):
            ids_list = []
            q_targ = calc_targ_from_aca(q_att, y_off, z_off)
            for ii, roll_offset in enumerate(roll_offsets):
                q_targ_roll = Quat([q_targ.ra, q_targ.dec, q_targ.roll + roll_offset])
                q_att_roll = calc_aca_from_targ(q_targ_roll, y_off, z_off)
                yag, zag = radec_to_yagzag(cands['ra'], cands['dec'], q_att_roll)
                row, col = yagzag_to_pixels(yag, zag, allow_bad=True, pix_zero_loc='edge')

                ok = (np.abs(row) < ACA.CCD['row_max']) & (np.abs(col) < ACA.CCD['col_max'])
                ids_list.append(set(cands['id'][ok]))
            return ids_list

        if roll_nom is None or roll_dev is None:
            import Ska.Sun
            pitch = Ska.Sun.pitch(q_att.ra, q_att.dec, self.date)
        if roll_nom is None:
            roll_nom = Ska.Sun.nominal_roll(q_att.ra, q_att.dec, self.date)
        if roll_dev is None:
            roll_dev = allowed_rolldev(pitch)

        # Ensure roll_nom in range 0 <= roll_nom < 360 to match q_att.roll
        roll_nom = roll_nom % 360.0
        roll_min = roll_nom - roll_dev
        roll_max = roll_nom + roll_dev

        # Get roll offsets spanning roll_min:roll_max with padding.  Padding
        # ensures that if a candidate is best at or beyond the extreme of
        # allowed roll then make sure the sampled rolls go out far enough so
        # that the mean of the roll_offset boundaries will get to the edge.
        roll = q_att.roll
        ro_minus = np.arange(0, roll_min - roll_dev - roll, -d_roll)[1:][::-1]
        ro_plus = np.arange(0, roll_max + roll_dev - roll, d_roll)
        roll_offsets = np.concatenate([ro_minus, ro_plus])

        # Get a list of the set of AGASC ids that are in the ACA FOV at each
        # roll offset.  Note that roll_offset is relative to roll (self.att[2])
        # and not roll_nom.
        ids_list = get_ids_list(roll_offsets)
        ids0 = ids_list[len(ro_minus)]

        # Get all unique sets of stars that are in the FOV over the sampled
        # roll offsets.  Ignore ids sets that do not add new candidate stars.
        uniq_ids_sets = []
        for ids in ids_list:
            if ids not in uniq_ids_sets and ids - ids0:
                uniq_ids_sets.append(ids)

        # print(f'Roll min, max={roll_min:.2f}, {roll_max:.2f}')
        # For each unique set, find the roll_offset range over which that set
        # is in the FOV.
        better_rolls = []
        for uniq_ids in uniq_ids_sets:
            # print(uniq_ids - ids0, ids0 - uniq_ids)
            # for sid in uniq_ids - ids0:
                # star = self.stars.get_id(sid)
                # print(f'{sid} {star["mag"]} {star["yang"]} {star["zang"]}')
            # This says that ``uniq_ids`` is a subset of available ``ids`` in
            # FOV for roll_offset.
            in_fov = np.array([uniq_ids <= ids for ids in ids_list])

            # Get the contiguous intervals where uniq_ids is in FOV
            intervals = logical_intervals(in_fov, x=roll_offsets + roll)

            for interval in intervals:
                # print(interval)
                if interval['x_start'] > roll_max or interval['x_stop'] < roll_min:
                    # print(f'skipping {roll_max} {roll_min}')
                    continue  # Interval completely outside allowed roll range
                better_roll = (interval['x_start'] + interval['x_stop']) / 2
                better_rolls.append(np.clip(better_roll, roll_min, roll_max))

        return sorted(set(better_rolls)), roll_min, roll_nom, roll_max

    def get_better_catalogs(self):

        if self.loud:
            print('  Exploring roll options')

        def improve_metric(n_stars, P2, n_stars_new, P2_new):
            n_stars_mult_x = np.array([2.0, 3.0, 4.0, 5.0])
            n_stars_mult_y = np.array([1.2, 0.6, 0.3, 0.15])

            P2_mult_x = np.array([1.0, 2.0, 3.0])
            P2_mult_y = np.array([2.0, 1.0, 0.5])

            n_stars_mult = np.interp(x=n_stars, xp=n_stars_mult_x, fp=n_stars_mult_y)
            P2_mult = np.interp(x=P2, xp=P2_mult_x, fp=P2_mult_y)
            dn = n_stars_rolled - n_stars
            dP2 = P2_rolled - P2
            n_stars_sign_mult = 2 if dn < 0 else 1
            P2_sign_mult = 2 if dP2 < 0 else 1
            out = (dn * n_stars_sign_mult * n_stars_mult +
                   dP2 * P2_sign_mult * P2_mult)
            return out

        P2 = -np.log10(self.acqs.calc_p_safe())
        n_stars = guide_count(self.guides['mag'], self.guides.t_ccd, self.is_ER)

        cand_idxs = self.get_candidate_better_stars()
        better_rolls, roll_min, roll_nom, roll_max = self.get_better_rolls(cand_idxs)

        q_att = Quat(self.att)
        q_targ = calc_targ_from_aca(q_att, 0, 0)

        better_acas = [deepcopy(self)]
        better_stats = [(P2, n_stars, 0.0)]

        for better_roll in better_rolls:
            q_targ_roll = Quat([q_targ.ra, q_targ.dec, better_roll])
            q_att_roll = calc_aca_from_targ(q_targ_roll, 0, 0)

            kwargs = self.call_args.copy()
            kwargs['att'] = q_att_roll

            aca_rolled = get_aca_catalog(**kwargs)

            P2_rolled = -np.log10(aca_rolled.acqs.calc_p_safe())
            n_stars_rolled = guide_count(aca_rolled.guides['mag'], aca_rolled.guides.t_ccd,
                                         self.is_ER)

            improvement = improve_metric(n_stars, P2,
                                         n_stars_rolled, P2_rolled)

            if improvement > 0.3:
                better_acas.append(aca_rolled)
                better_stats.append((P2_rolled, n_stars_rolled, improvement))

        better_stats = Table(rows=better_stats,
                             names=['P2', 'n_stars', 'improvement'],
                             meta={'roll_min': roll_min,
                                   'roll_max': roll_max,
                                   'roll_nom': roll_nom})
        return better_acas, better_stats
