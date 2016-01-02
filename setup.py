from setuptools import setup
from setuptools import find_packages

setup(name='Kerosene',
      version='0.3.0',
      description='Wrapper and downloader for fuel datasets',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/kerosene',
      download_url='https://github.com/dribnet/kerosene/tarball/0.3.0',
      license='MIT',
      install_requires=['fuel'],
      packages=find_packages())
