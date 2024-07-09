from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="cross_modality_data_augmentation",
    version="1.0",
    description="A robust, cross-modality-wise Data Augmentation technique capable of synthesizing new medical images from a given to a desired domain.",
    author="Julius Stutz",
    author_email="julius.stutz@stud.uni-bamberg.de",
    url="https://github.com/juliustutz00/cross_modality_data_augmentation",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)