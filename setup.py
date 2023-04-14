from setuptools import setup
from setuptools import find_packages

try:
    with open("requirements.txt") as f:
        requirements = f.readlines()
except:
    requirements = []

setup(
    name="KSolver",
    description="Kirchhoff solver for liquid's network",
    version="0.4.02",
    packages=find_packages(),
    url="https://github.com/ISGNeuroTeam/KirchhoffSolver",
    install_requires=requirements,
    options={"bdist_wheel": {"universal": 1}},
)
