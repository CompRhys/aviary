from setuptools import find_namespace_packages, setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="aviary",
    version="0.0.2",
    author="Rhys Goodall",
    author_email="rhys.goodall@outlook.com",
    description="A Collection of Machine Learning Models for Materials Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["aviary*"]),
    classifiers=[
        "Programming Language :: Python :: 3.7.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
