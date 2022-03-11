from setuptools import find_namespace_packages, setup

setup(
    name="aviary",
    version="0.0.3",
    author="Rhys Goodall",
    author_email="rhys.goodall@outlook.com",
    url="https://github.com/CompRhys/aviary",
    description="A Collection of Machine Learning Models for Materials Discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["aviary*"]),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "Material Science",
        "Machine Learning",
        "Wyckoff positions",
        "Crystal Structure Prediction",
    ],
    package_data={"": ["**/*.json"]},
    python_requires=">=3.7",
    install_requires=[
        "scipy",
        "tqdm",
        "torch",
        "numpy",
        "pymatgen",
        "torch_scatter",
        "pandas",
        "scikit_learn",
        "tensorboard",
        "typing_extensions;python_version<'3.8'",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov", "matminer"],
    },
)
