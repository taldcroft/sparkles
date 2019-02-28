# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Roll optimization during preliminary review of ACA catalogs.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import warnings

from astropy.table import Table, vstack
from chandra_aca.star_probs import acq_success_prob, guide_count
from chandra_aca.transform import (radec_to_yagzag, yagzag_to_pixels,
                                   calc_aca_from_targ, calc_targ_from_aca,
                                   snr_mag_for_t_ccd)
from Quaternion import Quat

from proseco.characteristics import CCD
from proseco import get_aca_catalog


ROLL_TABLE = Table.read(str(Path(__file__).parent / 'pitch_rolldev.csv'), format='ascii.basic',
                        guess=False, delimiter=',')


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
        in_fov = ((np.abs(stars['row']) < CCD['row_max'] - rc_pad) &
                  (np.abs(stars['col']) < CCD['col_max'] - rc_pad))
        radius2 = stars['row'] ** 2 + stars['col'] ** 2
        sp_ok = ~in_fov & (radius2 < 2 * (512 + rc_pad) ** 2)

        # Find potential acq stars that are noticably better than worst acq
        # star via the p_acq_model metric, defined as p_acq is at least 0.3
        # better.
        acq_ok = self.acqs.get_candidates_mask(stars)
        idxs = np.flatnonzero(sp_ok & acq_ok)
        p_acqs = acq_success_prob(date=self.acqs.date, t_ccd=self.acqs.t_ccd,
                                  mag=stars['mag'][idxs], color=stars['COLOR1'][idxs],
                                  spoiler=False, halfwidth=120)
        worst_p_acq = min(acq['probs'].p_acq_model(120) for acq in self.acqs)
        ok = p_acqs > worst_p_acq + 0.3
        better_acq_idxs = idxs[ok]

        # Find potential guide stars that are noticably better than worst guide
        # star, defined as being at least 0.2 mag brighter.
        guide_ok = self.guides.get_candidates_mask(stars)
        idxs = np.flatnonzero(sp_ok & guide_ok)
        worst_mag = np.max(self.guides['mag'])
        ok = stars['mag'][idxs] < worst_mag - 0.2
        better_guide_idxs = idxs[ok]

        # Take the union of better stars and return the indexes
        return sorted(set(better_guide_idxs) | set(better_acq_idxs))

    def _calc_targ_from_aca(self, q_att, y_off, z_off):
        """Wrapper around calc_tar_from_aca that is a no-op for ERs"""
        q_out = calc_targ_from_aca(q_att, y_off, z_off) if self.is_OR else q_att
        return q_out

    def _calc_aca_from_targ(self, q_att, y_off, z_off):
        """Wrapper around calc_aca_from_targ that is a no-op for ERs"""
        q_out = calc_aca_from_targ(q_att, y_off, z_off) if self.is_OR else q_att
        return q_out

    def get_roll_intervals(self, cand_idxs, roll_nom=None, roll_dev=None,
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

        att_targ = self.att_targ

        def get_ids_list(roll_offsets):
            ids_list = []

            for ii, roll_offset in enumerate(roll_offsets):
                # Roll about the target attitude, which is offset from ACA attitude by a bit
                att_targ_rolled = Quat([att_targ.ra, att_targ.dec, att_targ.roll + roll_offset])

                # Transform back to ACA pointing for computing star positions.
                att_rolled = self._calc_aca_from_targ(att_targ_rolled, y_off, z_off)

                # Get yag/zag row/col for candidates
                yag, zag = radec_to_yagzag(cands['ra'], cands['dec'], att_rolled)
                row, col = yagzag_to_pixels(yag, zag, allow_bad=True, pix_zero_loc='edge')

                ok = (np.abs(row) < CCD['row_max']) & (np.abs(col) < CCD['col_max'])
                ids_list.append(set(cands['id'][ok]))
            return ids_list

        # If roll_nom and roll_dev not supplied (which is normally the case) compute
        # them using Sun position.  Here we use the ACA attitude to get pitch since that
        #  is the official "spacecraft" attitude.
        att = self.att
        if roll_nom is None or roll_dev is None:
            import Ska.Sun
            pitch = Ska.Sun.pitch(att.ra, att.dec, self.date)
        if roll_nom is None:
            roll_nom = Ska.Sun.nominal_roll(att.ra, att.dec, self.date)
            att_nom = Quat([att.ra, att.dec, roll_nom])
            att_nom_targ = self._calc_targ_from_aca(att_nom, y_off, z_off)
            roll_nom = att_nom_targ.roll
        if roll_dev is None:
            roll_dev = allowed_rolldev(pitch)

        # Ensure roll_nom in range 0 <= roll_nom < 360 to match q_att.roll.
        # Also ensure that roll_min < roll < roll_max.  It can happen that the
        # ORviewer scheduled roll is outside the allowed_rolldev() range.  For
        # far-forward sun, allowed_rolldev() = 0.0.
        roll = att_targ.roll
        roll_nom = roll_nom % 360.0
        roll_min = min(roll_nom - roll_dev, roll - 0.1)
        roll_max = max(roll_nom + roll_dev, roll + 0.1)

        # Get roll offsets spanning roll_min:roll_max with padding.  Padding
        # ensures that if a candidate is best at or beyond the extreme of
        # allowed roll then make sure the sampled rolls go out far enough so
        # that the mean of the roll_offset boundaries will get to the edge.
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
        roll_intervals = []
        for uniq_ids in uniq_ids_sets:
            # print(uniq_ids - ids0, ids0 - uniq_ids)
            # for sid in uniq_ids - ids0:
                # star = self.stars.get_id(sid)
                # print(f'{sid} {star["mag"]} {star["yang"]} {star["zang"]}')

            # This says that ``uniq_ids`` is a subset of available ``ids`` in
            # FOV for roll_offset in the list comprehension below.  So everywhere
            # this list is True corresponds to a roll_offset where all the
            # ``uniq_ids`` are in the FOV.
            in_fov = np.array([uniq_ids <= ids for ids in ids_list])

            # Get the contiguous intervals of roll_offset where uniq_ids is in FOV
            intervals = logical_intervals(in_fov, x=roll_offsets + roll)

            for interval in intervals:
                if interval['x_start'] > roll_max or interval['x_stop'] < roll_min:
                    continue  # Interval completely outside allowed roll range

                roll_interval = {'roll': (interval['x_start'] + interval['x_stop']) / 2,
                                 'roll_min': interval['x_start'],
                                 'roll_max': interval['x_stop'],
                                 'add_ids': uniq_ids - ids0,
                                 'drop_ids': ids0 - uniq_ids}

                # Clip roll values to allowed range for obsid
                for key in ('roll', 'roll_min', 'roll_max'):
                    roll_interval[key] = np.clip(roll_interval[key], roll_min, roll_max)

                roll_intervals.append(roll_interval)

        roll_info = {'roll_min': roll_min,
                     'roll_max': roll_max,
                     'roll_nom': roll_nom}

        return sorted(roll_intervals, key=lambda x: x['roll']), roll_info

    def get_roll_options(self):

        if self.loud:
            print('  Exploring roll options')

        if self.roll_options is not None:
            warnings.warn('roll_options already available, not re-computing')
            return

        def improve_metric(n_stars, P2, n_stars_new, P2_new):
            """Ad-hoc metric defining improvement of a catalog.

            :param n_stars: original n_stars
            :param P2: original P2
            :param n_stars_new: new n_stars
            :param P2_new: new P2
            :returns: metric
            """
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
        roll_intervals, self.roll_info = self.get_roll_intervals(cand_idxs)

        att_targ = self.att_targ

        # Special case, first roll option is self but with obsid set to roll
        acar = deepcopy(self)
        acar.check_catalog()
        acar.is_roll_option = True
        roll_options = [{'acar': acar,
                         'P2': P2,
                         'n_stars': n_stars,
                         'improvement': 0.0,
                         'roll': att_targ.roll,
                         'roll_min': att_targ.roll,
                         'roll_max': att_targ.roll,
                         'add_ids': set(),
                         'drop_ids': set()}]

        for roll_interval in roll_intervals:
            roll = roll_interval['roll']
            att_targ_rolled = Quat([att_targ.ra, att_targ.dec, roll])
            att_rolled = self._calc_aca_from_targ(att_targ_rolled, 0, 0)

            kwargs = self.call_args.copy()

            # For roll optimization throw away the include/excludes
            for k1 in ('include', 'exclude'):
                for k2 in ('ids', 'halfws'):
                    for k3 in ('acq', 'guide'):
                        key = f'{k1}_{k2}_{k3}'
                        if key in kwargs:
                            del kwargs[key]

            kwargs['att'] = att_rolled

            aca_rolled = get_aca_catalog(**kwargs)

            P2_rolled = -np.log10(aca_rolled.acqs.calc_p_safe())
            n_stars_rolled = guide_count(aca_rolled.guides['mag'], aca_rolled.guides.t_ccd,
                                         count_9th=self.is_ER)

            improvement = improve_metric(n_stars, P2, n_stars_rolled, P2_rolled)

            if improvement > 0.3:
                acar = self.__class__(aca_rolled, obsid=self.obsid,
                                      is_roll_option=True)

                # Do the review and set up messages attribute
                acar.check_catalog()

                roll_option = {'acar': acar,
                               'P2': P2_rolled,
                               'n_stars': n_stars_rolled,
                               'improvement': improvement}
                roll_option.update(roll_interval)
                roll_options.append(roll_option)

        self.roll_options = roll_options
