# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Preliminary review of ACA catalogs selected by proseco.
"""
import io
import re
import weakref
from pathlib import Path
import pickle
from itertools import combinations, chain

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Quaternion import Quat
from jinja2 import Template
from chandra_aca.transform import yagzag_to_pixels, mag_to_count_rate
from astropy.table import Column, Table

import proseco
from proseco.catalog import ACATable
import proseco.characteristics as ACA

from . import test as aca_preview_test
from .roll_optimize import RollOptimizeMixin, guide_count

CACHE = {}
ACA_PREVIEW_VERSION = aca_preview_test(get_version=True)
PROSECO_VERSION = proseco.test(get_version=True)
FILEDIR = Path(__file__).parent


def main(sys_args=None):
    """Command line interface to preview_load()"""

    import argparse
    parser = argparse.ArgumentParser(
        description=f'ACA preliminary review tool {ACA_PREVIEW_VERSION}')
    parser.add_argument('load_name',
                        type=str,
                        help='Load name (e.g. JAN2119A) or full file name')
    parser.add_argument('--outdir',
                        type=str,
                        help='Output directory (default=<load name>')
    parser.add_argument('--report-level',
                        type=str,
                        default='none',
                        help="Make reports for messages at/above level "
                             "('all'|'none'|'info'|'caution'|'warning'|'critical') "
                             "(default='warning')")
    parser.add_argument('--roll-level',
                        type=str,
                        default='none',
                        help="Make alternate roll suggestions for messages at/above level "
                             "('all'|'none'|'info'|'caution'|'warning'|'critical') "
                             "(default='critical')")
    parser.add_argument('--obsid',
                        action='append',
                        help="Process only this obsid (can specify multiple times, default=all")
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Run quietly')
    args = parser.parse_args(sys_args)

    run_aca_review(args.load_name, outdir=args.outdir,
                   loud=(not args.quiet), report_level=args.report_level,
                   roll_level=args.roll_level, obsids=args.obsid)


def run_aca_review(load_name=None, *, acas=None, make_html=True, outdir=None,
                   report_level='none', roll_level='none', loud=False,
                   obsids=None, is_ORs=None):
    """Do ACA load review based on proseco pickle file from ORviewer.

    The ``load_name`` specifies the pickle file from which the ``ACATable``
    catalogs and obsids are read (unless ``acas`` and ``obsids`` are explicitly
    provided).  The following options are tried in this order:

    - <load_name> (e.g. 'JAN2119A_proseco.pkl')
    - <load_name>_proseco.pkl (for <load_name> like 'JAN2119A', ORviewer default)
    - <load_name>.pkl

    Instead of reading from a pickle, one can directly provide the catalogs as
    ``acas``.  In this case the ``load_name`` will only be used in the report
    HTML.

    When reading from a pickle, the ``obsids`` argument can be used to limit
    the list of obsids being processed.  This is handy for development or
    for examining just one obsid.

    If ``outdir`` is not provided then it will be set to ``load_name``.

    The ``report_level`` arg specifies the message category at which the full
    HTML report for guide and acquisition will be generated for obsids with at
    least one message at or above that level.  The options correspond to
    standard categories "info", "caution", "warning", and "critical".  The
    default is "none", meaning no reports are generated.  A final option is
    "all" which generates a report for every obsid.

    :param load_name: name of loads
    :param acas: list of ACAReviewTable objects (optional)
    :param make_html: make HTML output report
    :param outdir: output directory
    :param report_level: report level threshold for generating acq and guide report
    :param roll_level: level threshold for suggesting alternate rolls
    :param loud: print status information during checking
    :param obsids: list of obsids for selecting a subset for review (mostly for debug)
    :param is_ORs: list of is_OR values (for roll options review page)

    """
    if acas is None:
        acas = get_acas_from_pickle(load_name, loud)

    if obsids:
        acas = [aca for aca in acas if aca.obsid in obsids]

    if not acas:
        raise ValueError('no catalogs founds (check obsid filtering?)')

    # Make output directory if needed
    if make_html:
        # Generate outdir from load_name if necessary
        if outdir is None:
            if not load_name:
                raise ValueError('load_name must be provided if outdir is not specified')
            outdir = re.sub(r'(_proseco)?.pkl', '', load_name) + '_aca_preview'
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    # Special case when running a set of rolls for one obsid for the roll
    # options page.  The obsid in this case is actually roll but need to get
    # the OR/ER status right.  Setting the roll attribute is just a hint for
    # report processing to use the word "roll" instead of "obsid".
    if is_ORs:
        for aca, is_OR in zip(acas, is_ORs):
            aca._is_OR = is_OR
            aca.roll = True

    # Do the pre-review for all the catalogs
    for aca in acas:
        if not isinstance(aca, ACAReviewTable):
            raise TypeError('input catalog for review must be an ACAReviewTable')

        if loud:
            print(f'Processing obsid {aca.obsid}')

        aca.messages.clear()
        aca.context.clear()

        aca.set_stars_and_mask()  # Not stored in pickle, need manual restoration
        aca.check_catalog()

        if roll_level == 'all' or aca.messages >= roll_level:
            aca.get_roll_options()  # sets roll_options, roll_info attributes

        if make_html:

            # Output directory for the main prelim review index.html and for this obsid.
            # Note that the obs{aca.obsid} is not flexible because it must match the
            # convention used in ACATable.make_report().  Oops.
            aca.preview_dir = Path(outdir)
            aca.obsid_dir = aca.preview_dir / f'obs{aca.obsid}'
            aca.obsid_dir.mkdir(parents=True, exist_ok=True)

            aca.make_starcat_plot()

            if report_level == 'all' or aca.messages >= report_level:
                try:
                    aca.make_report()
                except Exception as err:
                    aca.add_message('critical', text=f'Running make_report() failed: {err}')

            if aca.roll_info:
                aca.make_roll_options_report()

            aca.context['text_pre'] = aca.get_text_pre()

    # noinspection PyDictCreation
    if make_html:
        context = {}
        context['load_name'] = load_name.upper()
        context['proseco_version'] = PROSECO_VERSION
        context['aca_preview_version'] = ACA_PREVIEW_VERSION
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


def get_acas_from_pickle(load_name, loud=False):
    """Get dict of proseco ACATable pickles for ``load_name``

    Note that ``load_name`` can be a Table with columns ``obsid``
    and ``aca`` (ACATable catalog object).

    :param load_name: load name or Table (see preview_load() doc for details)
    :param loud: print processing information
    """
    filenames = [load_name, f'{load_name}_proseco.pkl', f'{load_name}.pkl']
    for filename in filenames:
        pth = Path(filename)
        if pth.exists() and pth.is_file() and pth.suffix == '.pkl':
            if loud:
                print(f'Reading pickle file {filename}')
            acas_dict = pickle.load(open(filename, 'rb'))
            break
    else:
        raise FileNotFoundError(f'no matching pickle file {filenames}')

    acas = [ACAReviewTable(aca, obsid=obsid, loud=loud)
            for obsid, aca in acas_dict.items()]
    return acas


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
        # Is this being generated for a roll options report?
        ident = 'ROLL' if hasattr(aca, 'roll') else 'OBSID'
        line = (f'<a href="#id{aca.obsid}">{ident} = {obsid_str}</a>{fill}'
                f' at {aca.date}   '
                f'{aca.acq_count:.1f} ACQ | {aca.guide_count:.1f} GUI |')

        # Warnings
        for category in reversed(MessagesList.categories):
            msgs = aca.messages == category
            if msgs:
                text = stylize(f' {category.capitalize()}: {len(msgs)}', category)
                line += text

        lines.append(line)

    return '\n'.join(lines)


class MessagesList(list):
    categories = ('all', 'info', 'caution', 'warning', 'critical', 'none')

    def __eq__(self, other):
        if isinstance(other, str):
            return [msg for msg in self if msg['category'] == other]
        else:
            return super().__eq__(other)

    def __ge__(self, other):
        if isinstance(other, str):
            other_idx = self.categories.index(other)
            return [msg for msg in self
                    if self.categories.index(msg['category']) >= other_idx]
        else:
            return super().__ge__(other)


class ACAReviewTable(ACATable, RollOptimizeMixin):
    # def add_review_methods(cls, aca, *, obsid=None, loud=False):

    def __init__(self, *args, **kwargs):
        """Init review methods and attrs in ``aca`` object *in-place*.

        - Change ``aca.__class__`` to ``cls``
        - Add ``context`` and ``messages`` properties.

        :param aca: ACATable object, modified in place.
        :param obsid: obsid (optional)
        :param loud: print processing status info (default=False)

        """
        # if data is None:
        #    raise ValueError(f'data arg must be set to initialize {self.__class__.__name__}')

        obsid = kwargs.pop('obsid', None)
        loud = kwargs.pop('loud', False)

        # Make a copy of input aca table along with a deepcopy of its meta.
        super().__init__(*args, **kwargs)

        # Same hack as in __setstate__ (unpickling), namely to force any AcqProbs
        # objects to have the correct weakref to the current self.acqs.  As a reminder
        # the AcqProbs objects are found in the 'probs' column of self.acqs.cand_acqs.
        # The self.acqs['probs'] objects are just refs to the cand_acqs['probs'] ones.
        #
        # The _base_repr_ method of the base class does some trickery to normally display
        # the "interesting" columns with default formatting applied. This actually creates
        # a temporary table of this class with the right columns, but without any meta.  In
        # that case self.acqs will be None.
        if self.acqs is not None:
            for probs in self.acqs.cand_acqs['probs']:
                probs.acqs = weakref.ref(self.acqs)

        # Add row and col columns from yag/zag, if not already there.
        self.add_row_col()

        self.context = {}  # Jinja2 context for output HTML review
        self.messages = MessagesList()  # Warning messages
        self.loud = loud
        self.roll_options = None
        self.roll_info = None
        self.preview_dir = None
        self.obsid_dir = None
        self.roll_options_table = None
        self.acq_count = None
        self._is_OR = None

        # Input obsid could be a string repr of a number that might have have
        # up to 2 decimal points.  This is the case when obsid is taken from the
        # ORviewer dict of ACATable pickles from prelim review.  Tidy things up
        # in these cases.
        if obsid is not None:
            f_obsid = round(float(obsid), 2)
            i_obsid = int(f_obsid)
            num_obsid = i_obsid if (i_obsid == f_obsid) else f_obsid

            self.obsid = num_obsid
            self.acqs.obsid = num_obsid
            self.guides.obsid = num_obsid
            self.fids.obsid = num_obsid

        # Compute guide count once for the record
        # TODO make this a property
        if self.guides is not None:
            self.guide_count = guide_count(self.guides['mag'], self.guides.t_ccd)
            if self.is_ER:
                self.guide_count_9th = guide_count(self.guides['mag'], self.guides.t_ccd,
                                                   self.is_ER)

        if 'mag_err' not in self.colnames and self.acqs is not None and self.guides is not None:
            # Add 'mag_err' column after 'mag' using 'mag_err' from guides and acqs
            mag_errs = {entry['id']: entry['mag_err'] for entry in chain(self.acqs, self.guides)}
            mag_errs = Column([mag_errs.get(id, 0.0) for id in self['id']], name='mag_err')
            self.add_column(mag_errs, index=self.colnames.index('mag') + 1)

        # Don't want maxmag column
        if 'maxmag' in self.colnames:
            del self['maxmag']

    def run_aca_review(self, *, report_dir=None, report_level='none', roll_level='none'):
        """Do aca review based for this catalog

        The ``report_level`` arg specifies the message category at which the full
        HTML report for guide and acquisition will be generated for obsids with at
        least one message at or above that level.  The options correspond to
        standard categories "info", "caution", "warning", and "critical".  The
        default is "none", meaning no reports are generated.  A final option is
        "all" which generates a report for every obsid.

        :param report_dir: output directory for report
        :param report_level: report level threshold for generating acq and guide report
        :param roll_level: level threshold for suggesting alternate rolls

        :returns: ACAReviewTable object
        """
        acas = [self]

        make_html = (report_dir is not None)

        # Do aca review checks and update acas[0] in place
        run_aca_review(acas=acas, make_html=make_html, outdir=report_dir,
                       report_level=report_level, roll_level=roll_level,
                       load_name=f'Obsid {self.obsid}',
                       loud=False)

    def review_status(self):
        if self.thumbs_up:
            status = 1
        elif self.thumbs_down:
            status = -1
        else:
            status = 0

        return status

    @property
    def thumbs_up(self):
        n_crit_warn = len(self.messages == 'critical') + len(self.messages == 'warning')
        return n_crit_warn == 0

    @property
    def thumbs_down(self):
        n_crit = len(self.messages == 'critical')
        return n_crit > 0

    @property
    def is_OR(self):
        """Return ``True`` if obsid corresponds to an OR."""
        if self._is_OR is None:
            self._is_OR = self.obsid < 38000
        return self._is_OR

    @property
    def is_ER(self):
        """Return ``True`` if obsid corresponds to an ER."""
        return not self.is_OR

    def make_report(self):
        """Make report for acq and guide.

        """
        if self.loud:
            print(f'  Creating HTML reports for obsid {self.obsid}')

        # Make reports in preview_dir/obs<obsid>/{acq,guide}/
        super().make_report(rootdir=self.preview_dir)

        # Let the jinja template know this has reports and set the correct
        # relative link from <preview_dir>/index.html to the reports directory
        # containing acq/index.html and guide/index.html files.
        self.context['reports_dir'] = self.obsid_dir.relative_to(self.preview_dir).as_posix()

    def set_stars_and_mask(self):
        """Set stars attribute for plotting.

        This includes compatibility code to deal with somewhat-broken behavior
        in 4.3.x where the base plot method is hard-coded to look at
        ``acqs.stars`` and ``acqs.bad_stars``.

        """
        # Get stars from AGASC and set ``stars`` attribute
        self.set_stars()

    def make_roll_options_report(self):
        """Make a summary table and separate report page for roll options.

        """
        # Note self.roll_options includes the originally-planned roll case
        # as the first row.
        opts = [opt.copy() for opt in self.roll_options]
        rolls = [Quat(opt['aca'].att).roll for opt in self.roll_options]
        acas = [ACAReviewTable(opt['aca'], obsid=roll)
                for opt, roll in zip(opts, rolls)]

        for roll, opt in zip(rolls, opts):
            opt['roll'] = roll
            del opt['aca']

        opts_table = Table(opts, names=['roll', 'P2', 'n_stars', 'improvement',
                                        'roll_min', 'roll_max', 'add_ids', 'drop_ids'])
        for col in opts_table.itercols():
            if col.dtype.kind == 'f':
                col.info.format = '.2f'
        self.roll_options_table = opts_table

        # Make a separate preview page for the roll options
        is_ORs = [aca.obsid < 38000 for aca in acas]
        rolls_dir = self.obsid_dir / 'rolls'
        run_aca_review(f'Obsid {self.obsid} roll options',
                       acas=acas, outdir=rolls_dir, is_ORs=is_ORs,
                       report_level='none', roll_level='none', loud=False)

        # Add in a column with summary of messages in roll options e.g.
        # critical: 2 warning: 1
        msgs_summaries = []
        for aca in acas:
            texts = []
            for category in reversed(MessagesList.categories):
                msgs = aca.messages == category
                if msgs:
                    text = stylize(f'{category.capitalize()}: {len(msgs)}', category)
                    texts.append(text)
            msgs_summaries.append(' '.join(texts))
        self.roll_options_table['warnings'] = msgs_summaries

        # Set context for HTML output
        rolls_index = self.obsid_dir.relative_to(self.preview_dir) / 'rolls' / 'index.html'
        io_html = io.StringIO()
        self.roll_options_table.write(
            io_html, format='ascii.html',
            htmldict={'table_class': 'table-striped',
                      'raw_html_cols': ['warnings'],
                      'raw_html_clean_kwargs': {'tags': ['span'],
                                                'attributes': ['class']}})
        htmls = [line.strip() for line in io_html.getvalue().splitlines()]
        htmls = htmls[htmls.index('<table class="table-striped">'):htmls.index('</table>') + 1]
        self.context['roll_options_table'] = '\n'.join(htmls)
        self.context['roll_options_index'] = rolls_index.as_posix()
        for key in ('roll_min', 'roll_max', 'roll_nom'):
            self.context[key] = f'{self.roll_info[key]:.2f}'

    def make_starcat_plot(self):
        """Make star catalog plot for this observation.

        """
        plotname = f'starcat.png'
        outfile = self.obsid_dir / plotname
        self.context['catalog_plot'] = outfile.relative_to(self.preview_dir).as_posix()

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
Predicted Guide CCD temperature (max): {self.guides.t_ccd:.1f}
Predicted Acq CCD temperature (init) : {self.acqs.t_ccd:.1f}"""

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

    def check_catalog(self):
        """Perform all star catalog checks.

        """
        for entry in self:
            entry_type = entry['type']
            is_guide = entry_type in ('BOT', 'GUI')
            is_acq = entry_type in ('BOT', 'ACQ')
            is_fid = entry_type == 'FID'

            if is_guide or is_fid:
                self.check_guide_fid_position_on_ccd(entry)

            if is_guide:
                star = self.guides.get_id(entry['id'])
                self.check_pos_err_guide(star)
                self.check_imposters_guide(star)
                self.check_too_bright_guide(star)

            if is_guide or is_acq:
                self.check_bad_stars(entry)

        self.check_guide_geometry()
        self.check_acq_p2()
        self.check_guide_count()

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
        # Shortcuts and translate y/z to yaw/pitch
        dither_guide_y = self.dither_guide.y
        dither_guide_p = self.dither_guide.z

        # Set "dither" for FID to be pseudodither of 5.0 to give 1 pix margin
        # Set "track phase" dither for BOT GUI to max guide dither over
        # interval or 20.0 if undefined.  TO DO: hand the guide guide dither
        dither_track_y = 5.0 if (entry['type'] == 'FID') else dither_guide_y
        dither_track_p = 5.0 if (entry['type'] == 'FID') else dither_guide_p

        row_lim = ACA.max_ccd_row - ACA.CCD['window_pad']
        col_lim = ACA.max_ccd_col - ACA.CCD['window_pad']

        def sign(axis):
            """Return sign of the corresponding entry value.  Note that np.sign returns 0
            if the value is 0.0, not the right thing here.
            """
            return -1 if (entry[axis] < 0) else 1

        track_lims = {'row': (row_lim - dither_track_y * ACA.ARC_2_PIX) * sign('row'),
                      'col': (col_lim - dither_track_p * ACA.ARC_2_PIX) * sign('col')}

        for axis in ('row', 'col'):
            track_delta = abs(track_lims[axis]) - abs(entry[axis])
            for delta_lim, category in ((3.0, 'critical'),
                                        (5.0, 'info')):
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

    def check_guide_count(self):
        """Check for sufficient guide star fractional count

        """
        obs_type = 'ER' if self.is_ER else 'OR'
        if self.is_ER and self.guide_count_9th < 3.0:
            self.add_message(
                'critical',
                f'{obs_type} count of 9th mag guide stars {self.guide_count_9th:.2f} < 3.0')

        count_lim = 4.0 if self.is_OR else 6.0
        if self.guide_count < count_lim:
            self.add_message(
                'critical',
                f'{obs_type} count of guide stars {self.guide_count:.2f} < {count_lim}')

    def check_pos_err_guide(self, star):
        """Warn on stars with larger POS_ERR (warning at 1" critical at 2")

        """
        agasc_id = star['id']
        idx = self.get_id(agasc_id)['idx']
        # POS_ERR is in milliarcsecs in the table
        pos_err = star['POS_ERR'] * 0.001
        for limit, category in ((2.0, 'critical'),
                                (1.25, 'warning')):
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

    def check_bad_stars(self, entry):
        """Check if entry (guide or acq) is in bad star set from proseco

        :param entry: ACAReviewTable row
        :return: None
        """
        if entry['id'] in ACA.bad_star_set:
            msg = f'Star {entry["id"]} is in proseco bad star set'
            self.add_message('critical', msg, idx=entry['idx'])


# Run from source ``python -m sparkles.preview <load_name> [options]``
if __name__ == '__main__':
    main()
