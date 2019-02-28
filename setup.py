import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sampnn",
    version="0.0.1",
    author="Rhys Goodall",
    author_email="reag2@cam.ac.uk",
    description="Structure Agnostic Message Passing Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/comprhys/sampnn",
    packages=['sampnn'],
    package_dir={'sampnn': 'sampnn'},
    # package_data={'sampnn': ['tables/*.dat']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)