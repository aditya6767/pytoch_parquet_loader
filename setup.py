from setuptools import setup, find_packages

version = open('currentversion.txt').read().strip()
requirements = open('requirements.txt').read().split('\n')

setup(
    name="pytorch_parquet_loader",
    version=version,
    description="A library for loading large Parquet files as an IterableDataset in PyTorch using Dask.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aditya Jaiswal",
    author_email="aditya.jaiswal@gmail.com",
    url="https://github.com/aditya6767/pytoch_parquet_loader",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    project_urls={
        "PyPI": "https://pypi.org/project/pytorch_parquet_loader/",
    },
)
