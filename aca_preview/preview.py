# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Preliminary review of ACA catalogs selected by proseco.
"""
import re
from pathlib import Path
import pickle
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Quaternion import Quat
from jinja2 import Template
from chandra_aca.transform import yagzag_to_pixels, mag_to_count_rate
from chandra_aca.star_probs import guide_count
from astropy.table import Column

import proseco
from proseco.catalog import ACATable
import proseco.characteristics as CHAR
import proseco.characteristics_guide as GUIDE

CACHE = {}
VERSION = proseco.test(get_version=True)
FILEDIR = Path(__file__).parent
CATEGORIES = ('critical', 'warning', 'caution', 'info')


# Fix characteristics compatibility issues between 4.3.x and 4.4+
if not hasattr(CHAR, 'CCD'):
    for attr in ('CCD', 'PIX_2_ARC', 'ARC_2_PIX'):
        setattr(CHAR, attr, getattr(GUIDE, attr))


def main(sys_args=None):
    """Command line interface to preview_load()"""

    import argparse
    parser = argparse.ArgumentParser(description='ACA preliminary review tool')
    parser.add_argument('load_name',
                        type=str,
                        help='Load name (e.g. JAN2119A) or full file name')
    parser.add_argument('--outdir',
                        type=str,
                        help='Output directory (default=<load name>')
    parser.add_argument('--report-level',
                        type=str,
                        default='warning',
                        help="Make reports for messages at/above level "
                             "('all'|'none'|'info'|'caution'|'warning'|'critical') "
                             "(default='warning')")
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Run quietly')
    args = parser.parse_args(sys_args)

    preview_load(args.load_name, outdir=args.outdir,
                 loud=(not args.quiet), report_level=args.report_level)


def preview_load(load_name, outdir=None, report_level='none', loud=False):
    """Do preliminary load review based on proseco pickle file from ORviewer.

    The ``load_name`` specifies the pickle file.  The following options are tried
    in this order:
    - <load_name> (e.g. 'JAN2119A_proseco.pkl')
    - <load_name>_proseco.pkl (for <load_name> like 'JAN2119A', ORviewer default)
    - <load_name>.pkl

    If ``outdir`` is not provided then it will be set to ``load_name``.

    The ``report_level`` arg specifies the message category at which the full
    HTML report for guide and acquisition will be generated for obsids with at
    least one message at or above that level.  The options correspond to
    standard categories "info", "caution", "warning", and "critical".  The
    default is "none", meaning no reports are generated.  A final option is
    "all" which generates a report for every obsid.

    :param load_name: Name of loads
    :param outdir: Output directory
    :param report_level: report level threshold for generating acq and guide report
    :param loud: Print status information during checking

    """
    if load_name in CACHE:
        acas_dict = CACHE[load_name]
    else:
        acas_dict = get_acas(load_name, loud)
        CACHE[load_name] = acas_dict

    # Make output directory if needed
    if outdir is None:
        outdir = re.sub(r'(_proseco)?.pkl', '', load_name)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Convert dict of ACATable to list of ACAPreviewTable with obsid set correctly
    acas = []
    for obsid, aca in acas_dict.items():
        # Change instance class ``aca`` to include all the review methods. This is legal!
        ACAReviewTable.add_review_methods(aca, obsid=obsid, loud=loud, preview_dir=outdir)
        acas.append(aca)

    # Do the pre-review for all the catalogs
    for aca in acas:
        if loud:
            print(f'Processing obsid {aca.obsid}')

        aca.set_stars_and_mask()  # Not stored in pickle, need manual restoration
        aca.preview()
        aca.make_report(report_level)

    context = {}
    context['load_name'] = load_name.upper()
    context['version'] = VERSION
    context['acas'] = acas
    context['summary_text'] = get_summary_text(acas)

    template_file = FILEDIR / 'index_template_preview.html'
    template = Template(open(template_file, 'r').read())
    out_html = template.render(context)

    out_filename = outdir / 'index.html'
    if loud:
        print(f'Writing output review file {out_filename}')
    with open(out_filename, 'w') as fh:
        fh.write(out_html)


def stylize(text, category):
    """Stylize ``text``.

    Currently ``category`` of critical, warning, caution, or info are supported
    in the CSS span style.

    """
    out = f'<span class="{category}">{text}</span>'
    return out


def get_acas(load_name, loud=False):
    """Get dict of proseco ACATable pickles for ``load_name``

    :param load_name: load name (see preview_load() doc for details)
    :param loud: print processing information
    """
    filenames = [load_name, f'{load_name}_proseco.pkl', f'{load_name}.pkl']
    for filename in filenames:
        pth = Path(filename)
        if pth.exists() and pth.is_file() and pth.suffix == '.pkl':
            if loud:
                print(f'Reading pickle file {filename}')
            acas = pickle.load(open(filename, 'rb'))
            return acas
    raise FileNotFoundError(f'no matching pickle file {filenames}')


def get_summary_text(acas):
    """Get summary text for all catalogs.

    This is like::

      Proseco version: 4.4-r528-e9d6c73

      OBSID = -3898   at 2019:027:21:58:37.828   7.8 ACQ | 5.0 GUI |
      OBSID = 21899   at 2019:028:01:17:39.066   7.8 ACQ | 5.0 GUI |

    :param acas: list of ACATable objects
    :returns: str summary text
    """
    obsid_strs = [str(aca.obsid) for aca in acas]
    max_obsid_len = max(len(obsid_str) for obsid_str in obsid_strs)
    lines = []
    for aca, obsid_str in zip(acas, obsid_strs):
        fill = " " * (max_obsid_len - len(obsid_str))
        line = (f'<a href="#obsid{aca.obsid}">OBSID = {obsid_str}</a>{fill}'
                f' at {aca.date}   '
                f'{aca.acq_count:.1f} ACQ | {aca.guide_count:.1f} GUI |')

        # Warnings
        for category in CATEGORIES:
            msgs = [msg for msg in aca.messages if msg['category'] == category]
            if msgs:
                text = stylize(f' {category.capitalize()}: {len(msgs)}', category)
                line += text

        lines.append(line)

    return '\n'.join(lines)


class ACAReviewTable(ACATable):
    @classmethod
    def add_review_methods(cls, aca, *, obsid=None, loud=False, preview_dir='.'):
        """Add review methods to ``aca`` object *in-place*.

        - Change ``aca.__class__`` to ``cls``
        - Add ``context`` and ``messages`` properties.

        :param aca: ACATable object, modified in place.
        :param obsid: obsid (optional)
        :param loud: print processing status info (default=False)

        """
        aca.__class__ = cls
        aca.add_row_col()
        aca.context = {}  # Jinja2 context for output HTML review
        aca.messages = []  # Warning messages
        aca.loud = loud

        # Input obsid could be a string repr of a number that might have have
        # up to 2 decimal points.  This is the case when obsid is taken from the
        # ORviewer dict of ACATable pickles from prelim review.  Tidy things up
        # in these cases.
        if obsid is not None:
            f_obsid = round(float(obsid), 2)
            i_obsid = int(f_obsid)
            num_obsid = i_obsid if (i_obsid == f_obsid) else f_obsid

            aca.obsid = num_obsid
            aca.acqs.obsid = num_obsid
            aca.guides.obsid = num_obsid
            aca.fids.obsid = num_obsid

        # Clean up some attributes so acq/guide report summary looks OK.  This should
        # be fixed upstream at some point.
        for obj in (aca.acqs, aca.guides):
            obj.att = [round(val, 6) for val in Quat(obj.att).equatorial]
            obj.dither.y = round(obj.dither.y, 2)
            obj.dither.z = round(obj.dither.z, 2)
            obj.t_ccd = round(obj.t_ccd, 2)
        aca.acqs.man_angle = round(obj.man_angle, 2)

        # Output directory for the main prelim review index.html and for this obsid.
        # Note that the obs{aca.obsid} is not flexible because it must match the
        # convention used in ACATable.make_report().  Oops.
        aca.preview_dir = Path(preview_dir)
        aca.obsid_dir = aca.preview_dir / f'obs{aca.obsid}'
        aca.obsid_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_OR(self):
        """Return ``True`` if obsid corresponds to an OR."""
        return abs(self.obsid) < 38000

    @property
    def is_ER(self):
        """Return ``True`` if obsid corresponds to an ER."""
        return not self.is_OR

    def make_report(self, report_level):
        """Optionally make report for acq and guide.

        """
        if report_level == 'none':
            return

        if report_level != 'all':
            categories = ['info', 'caution', 'warning', 'critical']
            idx = categories.index(report_level)
            for category in categories[idx:]:
                msgs = [msg for msg in self.messages if msg['category'] == category]
                if msgs:
                    break
            else:
                # No messages at or above required level
                return

        if self.loud:
            print(f'  Creating HTML reports for obsid {self.obsid}')

        # Make reports in preview_dir/obs<obsid>/{acq,guide}/
        super().make_report(rootdir=self.preview_dir)

        # Let the jinja template know this has reports and set the correct
        # relative link from <preview_dir>/index.html to the reports directory
        # containing acq/index.html and guide/index.html files.
        self.context['reports_dir'] = self.obsid_dir.relative_to(self.preview_dir)

    def set_stars_and_mask(self):
        """Set stars attribute for plotting.

        This includes compatibility code to deal with somewhat-broken behavior
        in 4.3.x where the base plot method is hard-coded to look at
        ``acqs.stars`` and ``acqs.bad_stars``.

        """
        # Get stars from AGASC and set ``stars`` attribute
        self.set_stars()

        # Compatibility for 4.3.x before #221 (Make plot() method behave
        # consistently and correctly)
        if not hasattr(self, 'bad_stars_mask'):
            acqs = self.acqs
            acqs.stars = self.stars
            _, acqs.bad_stars = acqs.get_acq_candidates(acqs.stars)

    def make_starcat_plot(self):
        """Make star catalog plot for this observation.

        """
        plotname = f'starcat.png'
        outfile = self.obsid_dir / plotname
        self.context['catalog_plot'] = outfile.relative_to(self.preview_dir)

        if outfile.exists():
            return

        fig = plt.figure(figsize=(4.5, 4))
        ax = fig.add_subplot(1, 1, 1)
        self.plot(ax=ax)
        plt.tight_layout()
        fig.savefig(str(outfile))
        plt.close(fig)

    def get_text_pre(self):
        """Get pre-formatted text for report.

        """
        P2 = -np.log10(self.acqs.calc_p_safe())
        att = Quat(self.att)
        self._base_repr_()  # Hack to set default ``format`` for cols as needed
        catalog = '\n'.join(self.pformat(max_width=-1))
        self.acq_count = np.sum(self.acqs['p_acq'])
        self.guide_count = guide_count(self.guides['mag'], self.guides.t_ccd)

        message_text = self.get_formatted_messages()

        text_pre = f"""\
{self.detector} SIM-Z offset: {self.sim_offset}
RA, Dec, Roll (deg): {att.ra:.6f} {att.dec:.5f} {att.roll:.5f}
Dither acq: Y_amp= {self.dither_acq.y:.1f}  Z_amp={self.dither_acq.z:.1f}
Dither gui: Y_amp= {self.dither_guide.y:.1f}  Z_amp={self.dither_guide.z:.1f}
Maneuver Angle: {self.man_angle:.2f}
Date: {self.date}

{catalog}

{message_text}\
Probability of acquiring 2 or fewer stars (10^-x): {P2:.2f}
Acquisition Stars Expected: {self.acq_count:.2f}
Guide Stars count: {self.guide_count:.2f}
Predicted Guide CCD temperature (max): {self.t_ccd_guide:.1f}
Predicted Acq CCD temperature (init) : {self.t_ccd_acq:.1f}"""

        return text_pre

    def get_formatted_messages(self):
        """Format message dicts into pre-formatted lines for the preview report.

        """
        lines = []
        for message in self.messages:
            category = message['category']
            idx_str = f"[{message['idx']}] " if ('idx' in message) else ''
            line = f">> {category.upper()}: {idx_str}{message['text']}"
            line = stylize(line, category)
            lines.append(line)

        out = '\n'.join(lines) + '\n\n' if lines else ''
        return out

    def add_row_col(self):
        """Add row and col columns if not present

        """
        if 'row' in self.colnames:
            return

        row, col = yagzag_to_pixels(self['yang'], self['zang'], allow_bad=True)
        index = self.colnames.index('zang') + 1
        self.add_column(Column(row, name='row'), index=index)
        self.add_column(Column(col, name='col'), index=index + 1)

    def preview(self):
        """Prelim review of ``self`` catalog.

        This is based on proseco and (possibly) other available products, e.g. the DOT.
        """
        self.make_starcat_plot()
        self.add_row_col()
        self.check_catalog()
        self.context['text_pre'] = self.get_text_pre()

    def check_catalog(self):
        """Perform all star catalog checks.

        """
        for entry in self:
            self.check_guide_fid_position_on_ccd(entry)
            if entry['id'] in self.guides['id']:
                guide_star = self.guides.get_id(entry['id'])
                self.check_pos_err_guide(guide_star)
                self.check_imposters_guide(guide_star)
                self.check_too_bright_guide(guide_star)

        self.check_guide_geometry()
        self.check_acq_p2()
        self.check_bright_guide_for_ers()
        self.check_enough_guide_for_ers()

    def check_guide_geometry(self):
        """Check for guide stars too tightly clustered.

        (1) Check for any set of n_guide-2 stars within 500" of each other.
        The nominal check here is a cluster of 3 stars within 500".  For
        ERs this check is very unlikely to fail.  For catalogs with only
        4 guide stars this will flag for any 2 nearby stars.

        This check will likely need some refinement.

        (2) Check for all stars being within 2500" of each other.

        """
        ok = np.in1d(self['type'], ('GUI', 'BOT'))
        guide_idxs = np.flatnonzero(ok)
        n_guide = len(guide_idxs)

        def dist2(g1, g2):
            out = (g1['yang'] - g2['yang']) ** 2 + (g1['zang'] - g2['zang']) ** 2
            return out

        # First check for any set of n_guide-2 stars within 500" of each other.
        min_dist = 500
        min_dist2 = min_dist ** 2
        for idxs in combinations(guide_idxs, n_guide - 2):
            for idx0, idx1 in combinations(idxs, 2):
                # If any distance in this combination exceeds min_dist then
                # the combination is OK.
                if dist2(self[idx0], self[idx1]) > min_dist2:
                    break
            else:
                # Every distance was too small, issue a warning.
                msg = f'Guide indexes {idxs} clustered within {min_dist}" radius'
                self.add_message('critical', msg)

        # Check for all stars within 2500" of each other
        min_dist = 2500
        min_dist2 = min_dist ** 2
        for idx0, idx1 in combinations(guide_idxs, 2):
            if dist2(self[idx0], self[idx1]) > min_dist2:
                break
        else:
            msg = f'Guide stars all clustered within {min_dist}" radius'
            self.add_message('warning', msg)

    def check_guide_fid_position_on_ccd(self, entry):
        """Check position of guide stars and fid lights on CCD.

        """
        entry_type = entry['type']

        # Shortcuts and translate y/z to yaw/pitch
        dither_guide_y = self.dither_guide.y
        dither_guide_p = self.dither_guide.z

        # Set "dither" for FID to be pseudodither of 5.0 to give 1 pix margin
        # Set "track phase" dither for BOT GUI to max guide dither over
        # interval or 20.0 if undefined.  TO DO: hand the guide guide dither
        dither_track_y = 5.0 if (entry_type == 'FID') else dither_guide_y
        dither_track_p = 5.0 if (entry_type == 'FID') else dither_guide_p

        row_lim = CHAR.max_ccd_row - CHAR.CCD['window_pad']
        col_lim = CHAR.max_ccd_col - CHAR.CCD['window_pad']

        def sign(axis):
            """Return sign of the corresponding entry value.  Note that np.sign returns 0
            if the value is 0.0, not the right thing here.
            """
            return -1 if (entry[axis] < 0) else 1

        track_lims = {'row': (row_lim - dither_track_y * CHAR.ARC_2_PIX) * sign('row'),
                      'col': (col_lim - dither_track_p * CHAR.ARC_2_PIX) * sign('col')}

        if entry_type in ('GUI', 'BOT', 'FID'):
            for axis in ('row', 'col'):
                track_delta = abs(track_lims[axis]) - abs(entry[axis])
                for delta_lim, category in ((2.5, 'critical'),
                                            (5.0, 'warning')):
                    if track_delta < delta_lim:
                        text = (f"Less than {delta_lim} pix edge margin {axis} "
                                f"lim {track_lims[axis]:.1f} "
                                f"val {entry[axis]:.1f} "
                                f"delta {track_delta:.1f}")
                        self.add_message(category, text, idx=entry['idx'])
                        break

    # TO DO: acq star position check:
    # For acq stars, the distance to the row/col padded limits are also confirmed,
    # but code to track which boundary is exceeded (row or column) is not present.
    # Note from above that the pix_row_pad used for row_lim has 7 more pixels of padding
    # than the pix_col_pad used to determine col_lim.
    # acq_edge_delta = min((row_lim - dither_acq_y / ang_per_pix) - abs(pixel_row),
    #                          (col_lim - dither_acq_p / ang_per_pix) - abs(pixel_col))
    # if ((entry_type =~ /BOT|ACQ/) and (acq_edge_delta < (-1 * 12))){
    #     push @orange_warn, sprintf "alarm [%2d] Acq Off (padded) CCD by > 60 arcsec.\n",i
    # }
    # elsif ((entry_type =~ /BOT|ACQ/) and (acq_edge_delta < 0)){
    #     push @{self->{fyi}},
    #                 sprintf "alarm [%2d] Acq Off (padded) CCD (P_ACQ should be < .5)\n",i
    # }

    def add_message(self, category, text, **kwargs):
        """Add message to internal messages list.

        :param category: message category ('info', 'caution', 'warning', 'critical')
        :param text: message text
        :param \**kwargs: other kwarg

        """
        message = {'text': text, 'category': category}
        message.update(kwargs)
        self.messages.append(message)

    def check_acq_p2(self):
        """Check acquisition catalog safing probability.

        """
        P2 = -np.log10(self.acqs.calc_p_safe())
        obs_type = 'OR' if self.is_OR else 'ER'
        P2_lim = 2.0 if self.is_OR else 3.0
        if P2 < P2_lim:
            self.add_message('critical', f'P2: {P2:.2f} less than {P2_lim} for {obs_type}')
        elif P2 < P2_lim + 1:
            self.add_message('warning', f'P2: {P2:.2f} less than {P2_lim + 1} for {obs_type}')

    def check_bright_guide_for_ers(self, n_bright_req=3, bright_lim=9.0):
        """Check for at least 3 guide stars brighter than 9th mag for ERs.

        """
        n_bright = np.count_nonzero(self.guides['mag'] < bright_lim)
        if self.is_ER and n_bright < n_bright_req:
            self.add_message(
                'critical', f'ER bright stars: only {n_bright} stars brighter than {bright_lim}')

    def check_enough_guide_for_ers(self, n_required=8):
        """Warn on ERs with fewer than n_required (8) guide stars.

        """
        if self.is_ER and len(self.guides) < n_required:
            self.add_message('critical', f'ER guide stars: only {len(self.guides)} stars')

    def check_pos_err_guide(self, star):
        """Warn on stars with larger POS_ERR (warning at 1" critical at 2")

        """
        agasc_id = star['id']
        idx = self.get_id(agasc_id)['idx']
        # POS_ERR is in milliarcsecs in the table
        pos_err = star['POS_ERR'] * 0.001
        for limit, category in ((2.0, 'critical'),
                                (1.0, 'warning')):
            if pos_err > limit:
                self.add_message(
                    category,
                    f'Guide star {agasc_id} POS_ERR {pos_err:.2f}, limit {limit} arcsec',
                    idx=idx)
                break

    def check_imposters_guide(self, star):
        """Warn on stars with larger imposter centroid offsets

        """
        # Borrow the imposter offset method from starcheck
        def imposter_offset(cand_mag, imposter_mag):
            """
            For a given candidate star and the pseudomagnitude of the brightest 2x2 imposter
            calculate the max offset of the imposter counts are at the edge of the 6x6
            (as if they were in one pixel).  This is somewhat the inverse of
            proseco.get_pixmag_for_offset .
            """
            cand_counts = mag_to_count_rate(cand_mag)
            spoil_counts = mag_to_count_rate(imposter_mag)
            return spoil_counts * 3 * 5 / (spoil_counts + cand_counts)

        agasc_id = star['id']
        idx = self.get_id(agasc_id)['idx']
        offset = imposter_offset(star['mag'], star['imp_mag'])
        for limit, category in ((4.0, 'critical'),
                                (2.5, 'warning')):
            if offset > limit:
                self.add_message(
                    category,
                    f'Guide star imposter offset {offset:.1f}, limit {limit} arcsec',
                    idx=idx)
                break

    def check_too_bright_guide(self, star):
        """Warn on guide stars that may be too bright.

        - Critical if MAG_ACA_ERR used in selection is less than 0.1
        - Critical if within 2 * mag_err of the hard 5.8 limit, warn within 3 * mag_err
        - Warning if brighter than 6.1 (should be double-checked in
          context of other candidates).

        """
        agasc_id = star['id']
        idx = self.get_id(agasc_id)['idx']
        mag_err = star['mag_err']
        mag_aca_err = star['MAG_ACA_ERR'] * 0.01
        for mult, category in ((2, 'critical'),
                               (3, 'warning')):
            if star['mag'] - (mult * mag_err) < 5.8:
                self.add_message(
                    category,
                    f'Guide star {agasc_id} within {mult}*mag_err of 5.8 '
                    f'(mag_err={mag_err:.2f})', idx=idx)
                break
        if (star['mag'] < 6.1) and (mag_aca_err < 0.1):
            self.add_message(
                'critical',
                f'Guide star {agasc_id} < 6.1 with small MAG_ACA_ERR={mag_aca_err:.2f}.  '
                f'Double check selection.',
                idx=idx)
        elif star['mag'] < 6.1:
            self.add_message(
                'warning',
                f'Guide star {agasc_id} < 6.1. Double check selection.', idx=idx)


# Run from source ``python -m aca_preview.preview <load_name> [options]``
if __name__ == '__main__':
    main()
