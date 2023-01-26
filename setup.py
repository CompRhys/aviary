from setuptools import find_packages, setup

setup(
    name="aviary",
    version="0.1.0",
    author="Rhys Goodall",
    author_email="rhys.goodall@outlook.com",
    url="https://github.com/CompRhys/aviary",
    description="A collection of machine learning models for materials discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["aviary*"]),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "Graph Neural Network",
        "Machine Learning",
        "Materials Science",
        "Materials Discovery",
        "Materials Informatics",
        "Roost",
        "Self-Attention",
        "Transformer",
        "Wren",
        "Wyckoff positions",
    ],
    # if any package at most 2 levels under the aviary namespace contains *.json files,
    # include them in the package
    package_data={"aviary": ["**/*.json", "**/**/*.json"]},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "pymatgen",
        "scikit_learn",
        "tensorboard",
        "torch_scatter",
        "torch",
        "tqdm",
        "wandb",
    ],
    extras_require={
        # matminer for loading training data used in testing
        "test": ["pytest", "pytest-cov", "matminer"],
    },
)
