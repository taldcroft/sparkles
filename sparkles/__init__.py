from pkg_resources import get_distribution

try:
    _dist = get_distribution(__name__)
    __version__ = _dist.version

    if not __file__.startswith(_dist.location):
        # get_distribution() found a different package from this file, must be in source repo
        from setuptools_scm import get_version
        __version__ = get_version(root='..', relative_to=__file__)

except Exception:
    import warnings
    warnings.warn('Failed to find a package version, using 0.0.0')
    __version__ = '0.0.0'


from .core import run_aca_review, ACAReviewTable  # noqa


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
