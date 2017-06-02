# Copyright 2017 Google
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
# limitations under the License.Copyright [yyyy] [name of copyright owner]

from distutils.core import setup

setup(
    name='gridsim',
    version='1.0.0',
    author='Orion Pritchard',
    author_email='orionp@google.com',
    packages=['gridsim', 'gridsim.test'],
    package_dir={'gridsim': 'gridsim'},
    package_data={'gridsim': ['data/profiles/profiles_*.csv', 'data/costs/*.csv']},
    scripts=[],
    url='http://www.google.com',
    license='LICENSE',
    description='Files for simulating costs of electrical grid.',
    long_description=open('README.txt').read(),
    include_package_data=True,
    install_requires=[
        'numpy >= 1.11.1',
        'pandas >= 0.18.1',
        'ortools >= 5.0.3919',
    ],
)
