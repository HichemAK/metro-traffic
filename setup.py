import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metro-traffic",
    version="0.0.1",
    author="AMMAR KHODJA Hichem | BOUDJENIBA Oussama",
    author_email="hichem5696@gmail.com",
    description="Predict future metro traffic with LSTM Attention models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shade22413/metro-traffic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
