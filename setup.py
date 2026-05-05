from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline characters
        requirements = [req.replace("\n", "") for req in requirements]

        # Handle the editable install flag
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='Demand Forecasting for Ola Bike Ride Requests Using Machine Learning and Temporal Features',
    version='0.0.1',
    author='Subham Das',
    author_email='fi.subham_das@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)