from setuptools import setup, find_packages

setup(
    name="jinyang",
    version="1.0",
    description="update greyscale support",
    packages=find_packages(),
     install_requires=[
        "opencv-python",
        "numpy",
        "torch",
        "torchvision",
        "matplotlib",
        "tqdm",
        "pillow",
        "pandas",
        "psutil",
        "matplotlib",
        "simplejson",
        "psutil",
        "sklearn",
        "augly",
    ],
    author="lab"
)