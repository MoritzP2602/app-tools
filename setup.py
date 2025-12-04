from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="app-tools",
    version="1.1.1",
    author="Moritz Pabst",
    author_email="moritz.pabst@stud.uni-goettingen.de",
    description="Tools for working with Apprentice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MoritzP2602/app-tools",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "app-tools-chi_squared=app_tools.scripts.chi_squared:main",
            "app-tools-combine_weights=app_tools.scripts.combine_weights:main",
            "app-tools-create_grid=app_tools.scripts.create_grid:main",
            "app-tools-merge_surrogates=app_tools.scripts.merge_surrogates:main",
            "app-tools-split_weights=app_tools.scripts.split_weights:main",
            "app-tools-write_weights=app_tools.scripts.write_weights:main",
        ],
    },
    scripts=[
        "app_tools/scripts/app-tools-prepare_runs",
        "app_tools/scripts/app-tools-yodamerge_runs",
        "app_tools/scripts/app-tools-yodamerge_directories",
    ],
    include_package_data=True,
)
