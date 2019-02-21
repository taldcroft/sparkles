from sparkles import __version__

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

entry_points = {'console_scripts': ['sparkles=sparkles.core:main']}

setup(name='sparkles',
      author='Tom Aldcroft',
      description='Sparkles ACA review package',
      author_email='taldcroft@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      entry_points=entry_points,
      packages=['sparkles', 'sparkles.tests'],
      package_data={'sparkles': ['index_template*.html', 'pitch_rolldev.csv']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
