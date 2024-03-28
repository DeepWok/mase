#! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import codecs
import fileinput
import shutil

from contextlib import contextmanager

from setuptools import setup

# Get the long description
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def build_package(pkg_name, pkg_version):

    with codecs.open(os.path.join(BASE_DIR, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name=pkg_name,
        version=pkg_version,
        description="A fake package to warn the user they are not installing the correct package.",
        long_description=long_description,

        # The project's main homepage.
        url='https://github.com/NVIDIA',
        download_url='https://github.com/NVIDIA',

        # Author details
        author="Kitmaker",
        author_email='kitmaker@nvidia.com',

        # maintainer Details
        maintainer='Kitmaker',
        maintainer_email='kitmaker@nvidia.com',

        # The licence under which the project is released
        license='Apache2',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Information Technology',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Image Recognition',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development :: Libraries',
            'Topic :: Utilities',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Environment :: Console',
            'Natural Language :: English',
            'Operating System :: OS Independent',
        ],
        platforms=["Linux"],
        keywords='nvidia, deep learning, machine learning, supervised learning, unsupervised learning, reinforcement learning, logging',
    )

if len(sys.argv) != 4:
    raise RuntimeError("Bad params")

action=sys.argv[1]
package_name=sys.argv[2]
package_version=sys.argv[3]

# Need to remove the custom options or distutils will get sad
sys.argv.remove(package_name)
sys.argv.remove(package_version)

if action == "sdist":

    def maybe_delete_file(filename):
        try:
            os.remove(filename)
        except FileNotFoundError:
            pass

    @contextmanager
    def setup_sdist_environment(package_name):

        temporary_files = ["README.rst", "ERROR.txt", "PACKAGE_NAME"]
        for file in temporary_files:
            maybe_delete_file(file)

        with open("PACKAGE_NAME", "w") as f:
            f.write(package_name)

        shutil.copyfile("README.rst.in", "README.rst")
        shutil.copyfile("ERROR.txt.in", "ERROR.txt")

        def replace_in_file(search_text, new_text, filename):
            with fileinput.input(filename, inplace=True) as f:
                for line in f:
                    new_line = line.replace(search_text, new_text)
                    print(new_line, end='')

        replace_in_file("<PACKAGE_NAME>", package_name.upper(), filename="README.rst")
        replace_in_file("==============", "=" * len(package_name), filename="README.rst")

        replace_in_file("<package_name>", package_name, filename="README.rst")
        replace_in_file("<package_name>", package_name, filename="ERROR.txt")

        yield

        for file in temporary_files:
            maybe_delete_file(file)

        shutil.rmtree("%s.egg-info" % package_name.replace("-", "_"))

    print(f"\n[*] Building {package_name} ...")

    with setup_sdist_environment(package_name):
        build_package(package_name, package_version)

else:
    raise RuntimeError(open("ERROR.txt", "r").read())
