from aca_preview import __version__

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='aca_preview',
      author='Tom Aldcroft',
      description='ACA prelim products review',
      author_email='taldcroft@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['aca_preview', 'aca_preview'],
      package_data={'aca_preview': ['index_template*.html']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
