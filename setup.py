from setuptools import setup
from setuptools import find_packages

setup(name='Kerosene',
      version='0.1.0',
      description='Keras style wrapper for fuel datasets',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/kerosene',
      download_url='https://github.com/dribnet/kerosene/tarball/0.1.0',
      license='MIT',
      install_requires=['keras', 'fuel'],
      packages=find_packages())
