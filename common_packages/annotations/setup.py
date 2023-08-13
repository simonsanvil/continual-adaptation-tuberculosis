from setuptools import setup, find_packages

requirements = open("requirements.txt").read().splitlines()

setup(
    name="annotations",
    version="0.1.0",
    description="A package to handle labeling tasks and annotations.",
    author="Simon E. Sanchez Viloria",
    author_email="simsanch@inf.uc3m.es",
    packages=find_packages(where="."),
    python_requires=">=3.7",
    install_requires=requirements,
)
