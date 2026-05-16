from setuptools import setup, find_packages
import re

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("hytank/__init__.py").read(),
)[0]

setup(
    name="hytank",
    version=__version__,
    description="Liquid hydrogen tank modeling tool",
    license="MIT License",
    packages=find_packages(include=["hytank*", "lngtank*"]),
    package_data={"hytank": ["H2_property_data/*_properties.txt"]},
    install_requires=[
        "CoolProp>=7.1",
        "numpy>=1.24",
        "scipy>=1.10.0",
        "openmdao>=3.25",
    ],
    extras_require={
        "test": ["pytest", "parameterized"],
        "plot": ["matplotlib"],
    },
)
