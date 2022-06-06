from setuptools import find_packages, setup

setup(
    name="aviary",
    version="0.0.3",
    author="Rhys Goodall",
    author_email="rhys.goodall@outlook.com",
    url="https://github.com/CompRhys/aviary",
    description="A Collection of Machine Learning Models for Materials Discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["aviary*"]),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "Material Science",
        "Machine Learning",
        "Wyckoff positions",
        "Crystal Structure Prediction",
        "Self-Attention",
        "Transformer",
    ],
    package_data={"": ["embeddings/**/*.json"]},
    python_requires=">=3.7",
    install_requires=[
        "tqdm",
        "torch",
        "numpy",
        "pymatgen",
        "torch_scatter",
        "pandas",
        "scikit_learn",
        "tensorboard",
        "typing_extensions;python_version<'3.8'",
        "protobuf<4.21.0",  # temporary fix this issue:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/13159
    ],
    extras_require={
        "test": ["pytest", "pytest-cov", "matminer"],
    },
)
