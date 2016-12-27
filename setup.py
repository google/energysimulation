from distutils.core import setup

setup(
    name='gridsim',
    version='0.1.1',
    author='Orion Pritchard',
    author_email='orionp@google.com',
    packages=['gridsim', 'gridsim.test'],
    scripts=[],
    url='http://www.google.com',
    license='LICENSE',
    description='Files for simulating costs of electrical grid.',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy >= 1.11.1',
        'pandas >= 0.18.1',
        'ortools >= 5.0.3919',
        'protobuf >= 3.0.0b2'
    ],
)
