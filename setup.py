from setuptools import setup, find_packages

setup(
    name="cpimpdh",
    version="0.1.0",
    description="cpIMPDH LSTM pipeline",
    packages=find_packages(include=["cpimpdh", "cpimpdh.*"]),
)
