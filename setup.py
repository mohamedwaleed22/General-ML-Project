from setuptools import find_packages, setup
from typing import List


trigger = '-e .'
def get_requirements(filePath: str)-> List[str]:
    """
    Returns a list of required libraries 
    """
    reqs = []
    with open(filePath) as f:
        reqs = f.readlines()
        reqs = [req.replace("\n", "") for req in reqs]

        if trigger in reqs:
            reqs.remove(trigger)
    return reqs

# Package Info
setup(
    name='ML-Proj',
    version='0.1',
    author='Mohamed Waleed',
    author_email='mohawaleeed2000@gmail.com',
    packages=find_packages(),
    installReq=get_requirements('requirements.txt')

)

# __