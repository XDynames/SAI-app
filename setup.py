from setuptools import setup, find_packages

setup(
    name="StomAI",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
