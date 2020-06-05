# Anomaly Detection Framework

_________________

[![PyPI version](https://badge.fury.io/py/anomaly-detection-framework.svg)](https://pypi.org/project/anomaly-detection-framework)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.org/project/anomaly-detection-framework)
_________________

Anomaly-Detection-Framework is a platform for Time Series Anomaly Detection Proplems. Give the data to platform get the Anomaly Labels with scheduled time perioeds. That is simple is that!!!
Anomaly-Detection-Framework unables to Data Science communities easy to detect abnormal values on a Time Series Data Set. It is a platform which can run on Docker containers as services or python by using its modules. It also has the web interface which allows us to train - prediction - parameter tuning jobs easly.

##### Key Features
-   **Web Interface**: 

    It is web interface which allows us to connect data source and execute Machine Learning processes. You may create tasks according to your data set. These tasks are train models, Anomaly Detection Prediction, Parameter Tunning For Train Models.
    - *Menu*
        - [Data Source Configuraitons](http://127.0.0.1:7002/configs)
        - [Crate Task](http://127.0.0.1:7002/configs)
        - [Job Center](http://127.0.0.1:7002/ml_execute)
        - [Dashboard](http//:127.0.0.1:7002/dashboard)
    
-   **Schedule Machine Learning Jobs**:

    Three main process are able to initialized from platform Train, prediction and Parameter Tuning. Each Process can be scheduled Daily, Monthly, Weekly Hourly, etc with given time. In addition to that, you may run your Machine Learning processes real time.
    
-   **Dashboard Visualization**:

    When your data have connec to a data source and assign Date Indicator, Model Dimensions (optional), and feature column from Create Task  Menu [Create Task](http//:127.0.0.1:7002/query?messages=True), you may see the dashboard from [Dashboard](http//:127.0.0.1:7002/dashboard) Menu.
    
-   **Data Sources**: 
    Here are the data source you can conect with your SQL queries:
    
    - Ms SQL Server
    - PostgreSQL
    - AWS RedShift
    - Google BigQuery
    - .csv
    - .json
    - pickle
    
- **Models**:

    There are 2 Anomaly Detection Algorithm and FBProphet Anomaly Detection Solution are running on the platform.
    - LSTM
    - FBProphet
    - Isolation Forest
    
- **API Services**:

    There are 4 Services run on the platform.
    - Machine Learning Schedule Services
    - LSTM Model Service
    - FbProphet Model Service
    - Isolation Foreset model Service
    

- **Docker Compose Integration (Beta)**:

    Threse 4 containers are running on containers as services.
    - ml_executor-services
    - model-services-iso_f
    - model-services-lstm
    - model-services-prophet

---
    
# Running Platform


**1. You Have to Specify you directory**


```
import anomaly_detection as ad_exec

ad = ad_exec.AnomalyDetection(path='./Desktop', environment='local)
ad.init(apis=None)

```

Once, you have assigned the path a folder called 'Anomaly_Detection_Framework' will be created. 
Trained models will be imported to models file. Logs for both 'ml_execute', 'model_iso_f, 'model_prophet' and 'model_lstm' of log files will be created at logs file. 
Your .csv, .json or.yaml data source file must be copied to the data folder which is at 'Anomaly_Detection_Framework' folder.

- ###### **anomaly_detection.AnomalyDetection**:
    
    - **AnomalyDetection**
    
        ***path :*** Location where you are willing to create models and prediction data set.

        ***enviroment :** local or docker
    
        ***host :*** local or docker 
    
        ***remove_existed :*** remove data from the location where you have entered as path.
    
        ***master_node :*** if False, you must enter the services of information manually (port host, etc.). This allows user to initialize each service on different location or server or worker.
        If master_node=False there will not be a web interface. Once you create a master node, in order to use these services, you have clerify these services on a it.
    
    - **init**
    
        This initialized the folders. Checks the available ports for services in range between 6000 - 7000. Updates the apis.yaml if it is necessary.
        
        ***apis :***
        ```
        {
         'model_iso_f': {'port': 6000, 'host': '127.0.0.1'},
         'model_lstm': {'port': 6001, 'host': '127.0.0.1'}
        }
        
        ```
****

**2. Run The PLatform**

```
ad.run_platform()
```
This process initializesthe platform. Once you have run the code below you may have seen the services are running. 
If you assign master_node=True you may use enter to web interface from [http://127.0.0.1:7002/](http://127.0.0.1:7002/). 
If 7002 port is used from another platform directly platform assgns +1 port. (7003,7004, 7005, ..)

**2. Data Source**

You can connecto to data source from [Data Source Configuraitons](http://127.0.0.1:7002/configs).
There is two option to connect to a data source. You can integrate on web interface or you can use AnomalyDetection method in order to able to connect a data source.


- **Connection PostgreSQL - MS  SQL**

![connection_postgre](https://user-images.githubusercontent.com/26736844/83358571-27ab5200-a37d-11ea-95b9-b91b1ee38269.gif)

****

- **Connection .csv - .json - .yaml**

![Screen Recording 2020-06-06 at 12 33 AM](https://user-images.githubusercontent.com/26736844/83924666-d1c81700-a78d-11ea-8972-d2c14682440d.gif)

****

- **Connection Google BigQuery**

![Screen Recording 2020-06-06 at 12 49 AM](https://user-images.githubusercontent.com/26736844/83925434-d1308000-a78f-11ea-8753-847e48f73ff7.gif)

****

- **Create Tasks**
 After Data Source is created

![create_tasks](https://user-images.githubusercontent.com/26736844/83358834-e320b600-a37e-11ea-91d7-a0dbb351ea91.gif)

- **Job Run**

![Screen Recording 2020-06-06 at 01 25 AM](https://user-images.githubusercontent.com/26736844/83927175-cc21ff80-a794-11ea-885b-e7ec5bd38097.gif)

