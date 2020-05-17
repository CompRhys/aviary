import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roost",
    version="0.0.1",
    author="Rhys Goodall",
    author_email="reag2@cam.ac.uk",
    description="Representation Learning from Stoichiometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/comprhys/roost",
    packages=['roost'],
    package_dir={'roost': 'roost'},
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)