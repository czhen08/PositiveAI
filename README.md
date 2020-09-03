Important Information
================================================
Before running the backend code, make sure the BERT model file has been added to the main directory.


Code files
================================================
In this section, we define the source code files that are included in this submission with a description of the files 

BERT_classifier.ipynb: The code for developing, training and testing the BERT classifier.

app.py: The the root of the backend Flask app, serves the BERT model to make predictions and defines the endpoints for the frontend. 

model.py: Contains the class of the BERT model. Called by the app.py to serve the BERT model.

connect_db.py: Handles all the connections to the Azure SQL Database, contains functions to retrieve or store data.

news_operations.py: Contains operations on news data, including collecting and parsing news urls to obtain required news data.

news_parser.py: Thien's news parser that contains special parsing rules for 8 news outlet, it is used to get a tidy and clear news body for the frontend display.

url_extractor.py: Extract internal URLs from the Positive News, the Good News Network and the Crime Online. Save the URLs into CSV files. This python script is the first part of code to build the news sentiment dataset.

dataset_builder.py: Read the extracted URLs from the CSV file, parse the URL using the Python newspaper module to get the article content, and save the dataset. This python script is the second part of code to build the news sentiment dataset.

news_collect.py: In this project, we use the AYLIEN News API to collect target news articles before doing the data analysis experiments. This script is to collect the news published in June. It is put here as an example to show how we collect historical news articles. We collect different types of news articles (e.g. collect news by topics/publish_time/..), but the corresponding scripts are very similar and hence are not included in this folder to avoid dupelication.

figure_plot.py: An example script to plot the bar chart we use in the report.


Missing supplementary file
================================================
In this section, we define and describe the additional file that are not provided with the submission due to size constraints.

bert_model.bin: the saved BERT model, it needs to be loaded before running the backend. It is not included due to the size constraints.

To get the model, please download it from: https://github.com/czhen08/PositiveAI


Running the code
================
In this section, we describe how to run the code to reproduce results.

Before running the code, make sure the BERT model file has been added to the main directory. Also ensure the configuration data of the Aylien News API is still valid.