from setuptools import setup, find_packages

setup(
    name="jinyang",
    version="1.0",
    description="update greyscale support",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
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
        # "python-magic",
        # "python-magic-bin",
        "fvcore",
        "pytorchvideo"
    ],
    author="lab"
)
