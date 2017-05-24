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

from setuptools import setup
import os

MAJOR = 0
MINOR = 4
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
ISRELEASED = True

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

write_version_py()

LONG_DESCRIPTION = open(os.path.join(setup_dir, "README.md")).read()
DESCRIPTION = ("Simple wrapper library to the Apache Arrow-based "
               "Feather File Format")

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

URL = 'http://github.com/wesm/feather'

setup(
    name="feather-format",
    packages=['feather'],
    version=VERSION,
    package_data={'feather': ['*.pxd', '*.pyx']},
    install_requires=['pyarrow>=0.4.0'],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license='Apache License, Version 2.0',
    classifiers=CLASSIFIERS,
    author="Wes McKinney",
    author_email="wesm@apache.org",
    url=URL
)
