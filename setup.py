from aca_preview import __version__

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

entry_points = {'console_scripts': ['aca_preview=aca_preview.preview:main']}

setup(name='aca_preview',
      author='Tom Aldcroft',
      description='ACA prelim products review',
      author_email='taldcroft@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      entry_points=entry_points,
      packages=['aca_preview', 'aca_preview'],
      package_data={'aca_preview': ['index_template*.html']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
