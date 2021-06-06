#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "scikit-image",
    "scipy",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "nibabel",
    "pytest>=3",
]

extras_requirements = {
    "cli": ["nibabel", "numpy", "pandas"],
}

setup(
    author="Jacob Reinhold",
    author_email="jcreinhold@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="metrics for evaluating lesion segmentations",
    entry_points={"console_scripts": ["lesion-metrics=lesion_metrics.cli:main"]},
    install_requires=requirements,
    extras_require=extras_requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="lesion, metrics, segmentation, mri",
    name="lesion_metrics",
    packages=find_packages(include=["lesion_metrics", "lesion_metrics.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/jcreinhold/lesion-metrics",
    version="0.1.3",
    zip_safe=False,
)
