import os
import glob
import shutil
import re
import sys
import platform
import subprocess
​
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
​
setup(name='flightgym',
      version='0.0.1',
      author="Yunlong Song",
      author_email='song@ifi.uzh.ch',
      description="Flightmare: A Quadrotor Simulator",
      long_description='',
      packages=[''],
      package_dir={'': './'},
      package_data={'': ['flightgym.cpython-36m-x86_64-linux-gnu.so']},
      zip_fase=True,
      url=None,
)
Collapse








