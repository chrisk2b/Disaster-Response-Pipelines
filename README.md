# Disaster-Response-Pipelines
This repo contains the Disaster Respone Pipleines project of the Data Science Nanodegree.
It uses data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. Figure Eight provides data that contains Disaster messages which are classified w.r.t. 36 different categories (several different categories can be attached to a single message). The project consists of a

 - ETL pipeline which extracts raw data from csv-files. As part of this pipeline, the raw data gets processed, transformed, cleaned and merged, 
 - a ML pipeline which uses the output of the ETL-pipeline to train a model and store it as a pickle file,
 - a Web App which provides an interface to classify new messages.

  

## Core Components
The major files in this repository are 

 - load_data.py: it is contained in the /data folder and loads the csv-files disaster_categories.csv and disaster_messages.csv which are contained in the same folder, merges these files, cleanse the data and stores the results in a SQLite database called disaster.db
 - train_classifier.py: this file is contained in the /models folder and
	 
	 - loads the processed data from the SQLite database data/disaster.db
	 - splits the data into training and testing sets
	 - trains a Random Forest classifier (with multiple outputs) on the training sets
	 - evaluates its performance on the test sets
	 - stores the trained model as a pickle file (model.pkl) in the models folder
	 
 - run.py: this file is contained in the app folder and is used to start a Flask Webb App. The app folder also contained the templates folder, which contains the htlm-files mater.html and go.html. The master.html file renders the wep page which is used to enter new disaster messages while the go.html file is used to rendes the classification results.


## Usage

 - To run the ETL pipeline that cleans the data and stores it in a database open a terminal and type `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster.db`
 - To train, evaluate and pickle the model, open a terminal an type
`python models/train_classifier.py data/disaster.db models/model.pkl`
 - To start the web app, open a terminal and run the following command in the app's directory:
    `python run.py`

Afterwards, the application can be used by using the url http://localhost:3001/
Once that url is called, the master.html page gets rendered and one can type in a new disaster message in the form. By pressing the button at that page, the go.html page will display the classification results.


