from setuptools import setup, find_packages

setup(
    name="jinyang",
    version="1.0",
    description="update greyscale support",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "torch==1.8.0+cu111",
        "torchvision==0.9.0+cu111",
        "torchaudio==0.8.0"
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
