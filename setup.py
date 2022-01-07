from setuptools import setup, find_packages

# import pathlib
# here = pathlib.Path(__file__).parent.resolve()
# # Get the long description from the README file
# long_description = (here / 'README.md').read_text(encoding='utf-8')

long_description = None

setup(
    name='plasx',
#    version='0.1',
    description='A plasmid sequence classifier',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/michaelkyu/plasx',
    author='Michael Ku Yu', 
    author_email='michaelkuyu@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='plasmid, metagenomics',
    packages=find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'numba',
        'blosc'], 
    entry_points={
        'console_scripts': [
            'plasx=plasx.plasx_script:run',
        ],
    },
    project_urls={
        'Source': 'https://github.com/michaelkyu/plasx/',
    },
)
