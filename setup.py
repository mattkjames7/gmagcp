import setuptools
from setuptools.command.install import install
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gmagcp",
    version="0.0.1",
    author="Matthew Knight James",
    author_email="mattkjames7@gmail.com",
    description="Ground magnetometer cross phase.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattkjames7/gmagcp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: POSIX",
    ],
    install_requires=[
		"numpy",
        "scipy",
		"matplotlib",
		"DateTimeTools",
		"wavespec>=0.0.4",
		"RecarrayTools",
		"PyFileIO",
        "groundmag"
	],
	include_package_data=True,
)



