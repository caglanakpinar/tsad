import setuptools
from setuptools import find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="anomaly_detection_framework",
    version="0.0.21",
    author="Caglan Akpinar",
    author_email="cakpinar23@gmail.com",
    description="Anomaly Detection Framework allows us to calculate Anomalities on any Time - Series Data Sets. It has an interface which is easy to manage to train - predict with given dataset.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='anomaly Time Series Anomaly LSTM Prophet Isolation Forest',
    packages= find_packages(exclude='__pycache__'),
    py_modules=['anomaly_detection', 'anomaly_detection/web'],
    install_requires=[
        "requests",
        "convertdate",
        "lunarcalendar",
        "holidays",
        "docker-compose >= 1.25.5",
        "numpy >= 1.18.1",
        "pandas >= 0.25.3",
        "scipy >= 1.4.1 ",
        "tensorflow >= 2.2.0",
        "PyYAML",
        "schedule >= 0.6.0",
        "scikit-learn >= 0.22.1",
        "DateTime>= 4.3",
        "Flask >= 1.1.1",
        "multiprocess >= 0.70.9",
        "google-cloud-bigquery",
        "mysql-connector-python",
        "plotly >=  4.5.0",
        "dash-html-components >= 1.0.2",
        "dash-core-components >=  1.8.0",
        "dash >= 1.9.0",
        "threaded >= 4.0.8",
        "requests >= 2.23.0",
        "pytest-shutil >= 1.7.0",
        "python-dateutil >= 2.8.1",
        "sockets >= 1.0.0",
        "random2 >= 1.0.1",
        "psycopg2 >= 2.8.5",
        "convertdate",
        "LunarCalendar",
        "holidays",
        "pystan",
        "fbprophet == 0.6",
        "Keras >= 2.3.1"
    ],
    url="https://github.com/caglanakpinar/tsad",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
