from setuptools import setup, find_packages

exec(open('plantcelltype/__version__.py').read())
setup(
    name='plantcelltype',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    description='Training environment for PlantCellType Graph Benchmark',
    author='Lorenzo Cerrone',
    url='https://github.com/hci-unihd/plant-celltype',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)
