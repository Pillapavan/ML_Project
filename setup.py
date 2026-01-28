from setuptools import setup, find_packages
from typing import List

hypen_e_dot = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []  
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return [req.replace("\n", "") for req in requirements]



setup(
    name='MlProject',
    version='0.1.0',
    packages=find_packages(),
    author='Pavan',
    author_email='pillapavan90909@gmail.com',
    install_requires=get_requirements('requirements.txt')
)