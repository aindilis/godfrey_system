from setuptools import setup, find_packages

setup(
    name="godfrey_system",
    version="1.0.0",
    description="A vigilance and strategic intelligence system inspired by Godfrey O'Donnell of Tyrconnell",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "godfrey=godfrey_system.main:main",
            "godfrey-simulator=godfrey_system.simulator:main",
        ],
    },
    python_requires=">=3.8",
)

