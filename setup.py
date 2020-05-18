import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anomaly_detection_framework",
    version="0.0.1",
    author="Caglan Akpinar",
    author_email="cakpinar23@gmail.com",
    description="Anomaly Detection Frameworks allows us to caluculate Anomalities on any Time - Series Data Sets. It has an interdace which easy to manage to train - predict given dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='anomaly Time Series Anomaly LSTM Prophet Isolation Forest',
    packages=["src", "platform", "services", "documents"],
    install_requires=[
        "docker-compose >= 1.25.5"
    ],
    url="https://github.com/caglanakpinar/tsad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)