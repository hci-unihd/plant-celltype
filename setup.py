from setuptools import setup, find_packages

exec(open('plantcelltype/__version__.py').read())
setup(
    name='plantcelltype',
    version=__version__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    description='Training environment for PlantCellType Graph Benchmark',
    author='Anonymous',
    url='TODO',
    author_email='TODO',
)
