import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trajectory-nearest-neighbors",
    version="1.0",
    author="Arnaud Pannatier",
    author_email="arnaud.pannatier@idiap.ch",
    description=
    "Find Nearest Neighbors efficiently if the data set is arranged in trajectories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="tobedetermined",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["torch>=1.8", "numpy", "pytest"],
    python_requires='>=3.6',
    license="GNU AGPLv3")
