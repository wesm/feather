#!/usr/bin/env python
#
# Copyright 2016 Feather Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Bits here from Apache Kudu (incubating), ASL 2.0

from Cython.Distutils import build_ext
from Cython.Build import cythonize
import Cython

import numpy as np

import sys
from setuptools import setup
from distutils.command.clean import clean as _clean
from distutils.extension import Extension
import os
import platform

if Cython.__version__ < '0.19.1':
    raise Exception('Please upgrade to Cython 0.19.1 or newer')

MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
ISRELEASED = False

setup_dir = os.path.abspath(os.path.dirname(__file__))


def write_version_py(filename=os.path.join(setup_dir, 'feather/version.py')):
    version = VERSION
    if not ISRELEASED:
        version += '.dev'

    a = open(filename, 'w')
    file_content = "\n".join(["",
                              "# THIS FILE IS GENERATED FROM SETUP.PY",
                              "version = '%(version)s'",
                              "isrelease = '%(isrelease)s'"])

    a.write(file_content % {'version': VERSION,
                            'isrelease': str(ISRELEASED)})
    a.close()


class clean(_clean):
    def run(self):
        _clean.run(self)
        for x in ['feather/ext.cpp']:
            try:
                os.remove(x)
            except OSError:
                pass

if 'FEATHER_HOME' in os.environ:
    prefix = os.environ['FEATHER_HOME']
    sys.stderr.write("Building from configured libfeather prefix {0}\n"
                     .format(prefix))
else:
    if os.path.exists("/usr/local/include/feather"):
        prefix = "/usr/local"
    elif os.path.exists("/usr/include/feather"):
        prefix = "/usr"
    else:
        sys.stderr.write("Cannot find installed libfeather core library.\n")
        sys.exit(1)
    sys.stderr.write("Building from system prefix {0}\n".format(prefix))
    feather_include_dir = prefix + "/include"
    feather_lib_dir = prefix + "/lib"

feather_include_dir = os.path.join(prefix, 'include')
feather_lib_dir = os.path.join(prefix, 'lib')
INCLUDE_PATHS = [feather_include_dir, 'feather', np.get_include()]
LIBRARY_DIRS = [feather_lib_dir]
RT_LIBRARY_DIRS = LIBRARY_DIRS
extra_link_args = []

if platform.system() == 'Darwin':
    extra_link_args.append('-Wl,-rpath,' + feather_lib_dir)

ext = Extension('feather.ext',
                ['feather/ext.pyx'],
                language='c++',
                libraries=['feather'],
                include_dirs=INCLUDE_PATHS,
                library_dirs=LIBRARY_DIRS,
                runtime_library_dirs=RT_LIBRARY_DIRS,
                extra_compile_args=['-std=c++11', '-O0'],
                extra_link_args=extra_link_args)
extensions = [ext]
extensions = cythonize(extensions)

write_version_py()

LONG_DESCRIPTION = open(os.path.join(setup_dir, "README.md")).read()
DESCRIPTION = "Python interface to the Apache Arrow-based Feather File Format"

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Cython'
]

setup(
    name="feather-python",
    packages=['feather', 'feather.tests'],
    version=VERSION,
    package_data={'feather': ['*.pxd', '*.pyx']},
    ext_modules=extensions,
    cmdclass={
        'clean': clean,
        'build_ext': build_ext
    },
    install_requires=['cython >= 0.21'],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license='Apache License, Version 2.0',
    classifiers=CLASSIFIERS,
    author="Wes McKinney",
    maintainer_email="wesm@apache.org",
    test_suite="feather.tests"
)
