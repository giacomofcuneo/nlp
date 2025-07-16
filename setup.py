from setuptools import setup, find_packages

setup(
    name='data4allnlp',
    version='0.1.0',
    author='Data4All Team',
    description='A nlp package for Data4All projects',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'torch>=2.0.0',
        'transformers>=4.40.0',
        'datasets',
        'pyyaml>=5.4.1',
    ],
    python_requires='>=3.10',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'operating system :: OS Independent',
        'License :: gnu :: gplv3',
    ],
)