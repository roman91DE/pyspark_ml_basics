#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMSkillsNetworkBD0231ENCoursera2789-2023-01-01">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 

# ## Final Project - Build an ML Pipeline for Airfoil noise prediction
# 

# Estimated time needed: **90** minutes
# 

# ## Scenario
# 

# You are a data engineer at an aeronautics consulting company. Your company prides itself in being able to efficiently design airfoils for use in planes and sports cars. Data scientists in your office need to work with different algorithms and data in different formats. While they are good at Machine Learning, they count on you to be able to do ETL jobs and build ML pipelines. In this project you will use the modified version of the NASA Airfoil Self Noise dataset. You will clean this dataset, by dropping the duplicate rows, and removing the rows with null values. You will create an ML pipe line to create a model that will predict the SoundLevel based on all the other columns. You will evaluate the model and towards the end you will persist the model.
# 
# 

# ## Objectives
# 
# In this 4 part assignment you will:
# 
# - Part 1 Perform ETL activity
#   - Load a csv dataset
#   - Remove duplicates if any
#   - Drop rows with null values if any
#   - Make transformations
#   - Store the cleaned data in parquet format
# - Part 2 Create a  Machine Learning Pipeline
#   - Create a machine learning pipeline for prediction
# - Part 3 Evaluate the Model
#   - Evaluate the model using relevant metrics
# - Part 4 Persist the Model 
#   - Save the model for future production use
#   - Load and verify the stored model
# 

# ## Datasets
# 
# In this lab you will be using dataset(s):
# 
#  - The original dataset can be found here NASA airfoil self noise dataset. https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
#  
#  - This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
# 

# Diagram of an airfoil. - For informational purpose
# 

# ![Airfoil with flow](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/images/Airfoil_with_flow.png)
# 

# Diagram showing the Angle of attack. - For informational purpose
# 

# ![Airfoil angle of attack](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/images/Airfoil_angle_of_attack.jpg)
# 

# ## Before you Start
# 

# **Before you start attempting this project it is highly recommended that you finish the practice project.**
# 

# ## Setup
# 

# For this lab, we will be using the following libraries:
# 
# *   [`PySpark`](https://spark.apache.org/docs/latest/api/python/index.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMSkillsNetworkBD0231ENCoursera2789-2023-01-01) for connecting to the Spark Cluster
# 

# ### Installing Required Libraries
# 
# Spark Cluster is pre-installed in the Skills Network Labs environment. However, you need libraries like pyspark and findspark to
#  connect to this cluster.
# 

# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:
# 

# In[1]:


get_ipython().system('pip install pyspark==3.1.2 -q')
get_ipython().system('pip install findspark -q')


# ### Importing Required Libraries
# 
# _We recommend you import all required libraries in one place (here):_
# 

# In[2]:


# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn
warnings.filterwarnings('ignore')

# FindSpark simplifies the process of using Apache Spark with Python

import findspark

findspark.init()


# ## Part 1 - Perform ETL activity
# 

# ### Task 1 - Import required libraries
# 

# In[4]:


from pyspark.sql import SparkSession

# ### Task 2 - Create a spark session
# 

# In[8]:


spark = (
    SparkSession
    .builder 
    .master("local")
    .appName("Word Count")
    .getOrCreate()
)


# ### Task 3 - Load the csv file into a dataframe
# 

# Download the data file.
# 
# NOTE : Please ensure you use the dataset below and not the original dataset mentioned above.
# 

# In[9]:


get_ipython().system('wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-BD0231EN-Coursera/datasets/NASA_airfoil_noise_raw.csv')


# In[14]:


get_ipython().system('head -n3 "NASA_airfoil_noise_raw.csv"')


# Load the dataset into the spark dataframe
# 

# In[20]:


# Load the dataset that you have downloaded in the previous task

from pyspark.sql.types import FloatType, StructField, StructType

schema = StructType([
    StructField("Frequency", FloatType(), True),
    StructField("AngleOfAttack", FloatType(), True),
    StructField("ChordLength", FloatType(), True),
    StructField("FreeStreamVelocity", FloatType(), True),
    StructField("SuctionSideDisplacement", FloatType(), True),
    StructField("SoundLevel", FloatType(), False)
])

df = spark.read.csv("NASA_airfoil_noise_raw.csv", schema=schema, header=True)



# ### Task 4 - Print top 5 rows of the dataset
# 

# In[21]:


df.show(5)


# ### Task 6 - Print the total number of rows in the dataset
# 

# In[23]:


rowcount1 = df.count()
print(rowcount1)


# ### Task 7 - Drop all the duplicate rows from the dataset
# 

# In[24]:


df = df.dropDuplicates()


# ### Task 8 - Print the total number of rows in the dataset
# 

# In[25]:


#your code goes here

rowcount2 = df.count()
print(rowcount2)


# ### Task 9 - Drop all the rows that contain null values from the dataset
# 

# In[26]:


df = df.dropna()


# ### Task 10 - Print the total number of rows in the dataset
# 

# In[28]:


#your code goes here

rowcount3 = df.count()
print(rowcount3)


# ### Task 11 - Rename the column "SoundLevel" to "SoundLevelDecibels"
# 

# In[29]:


df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels")


# ### Task 12 - Save the dataframe in parquet format, name the file as "NASA_airfoil_noise_cleaned.parquet"
# 

# In[31]:


df.write.parquet("NASA_airfoil_noise_cleaned.parquet", mode="overwrite", compression="snappy")


# #### Part 1 - Evaluation
# 

# 
# **Run the code cell below.**<br>
# **Use the answers here to answer the final evaluation quiz in the next section.**<br>
# **If the code throws up any errors, go back and review the code you have written.**</b>
# 

# In[34]:


print("Part 1 - Evaluation")

print("Total rows = ", rowcount1)
print("Total rows after dropping duplicate rows = ", rowcount2)
print("Total rows after dropping duplicate rows and rows with null values = ", rowcount3)
print("New column name = ", df.columns[-1])

import os

print("NASA_airfoil_noise_cleaned.parquet exists :", os.path.isdir("NASA_airfoil_noise_cleaned.parquet"))


# ## Part - 2 Create a  Machine Learning Pipeline
# 

# ### Task 1 - Load data from "NASA_airfoil_noise_cleaned.parquet" into a dataframe
# 

# In[35]:


#your code goes here

df = spark.read.parquet("NASA_airfoil_noise_cleaned.parquet")


# ### Task 2 - Print the total number of rows in the dataset
# 

# In[36]:


#your code goes here

rowcount4 = df.count()
print(rowcount4)



# In[37]:


df.printSchema()


# ### Task 3 - Define the VectorAssembler pipeline stage
# 

# Stage 1 - Assemble the input columns into a single column "features". Use all the columns except SoundLevelDecibels as input features.
# 

# In[38]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=[
        "Frequency", "AngleOfAttack", "ChordLength", "FreeStreamVelocity", "SuctionSideDisplacement"
    ],
    outputCol="vectorized",
    handleInvalid='error'
)



# ### Task 4 - Define the StandardScaler pipeline stage
# 

# Stage 2 - Scale the "features" using standard scaler and store in "scaledFeatures" column
# 

# In[40]:


from pyspark.ml.feature import StandardScaler

scaler = StandardScaler(
    withMean=False,
    withStd=True,
    inputCol="vectorized",
    outputCol="features"
)


# ### Task 5 - Define the Model creation pipeline stage
# 

# Stage 3 - Create a LinearRegression stage to predict "SoundLevelDecibels"
# 
# **Note:You need to use the scaledfeatures retreived in the previous step(StandardScaler pipeline stage).**
# 

# In[41]:


from pyspark.ml.regression import LinearRegression

lr = LinearRegression(
    featuresCol='features',
    labelCol='SoundLevelDecibels',
    predictionCol='prediction'
)


# ### Task 6 - Build the pipeline
# 

# Build a pipeline using the above three stages
# 

# In[42]:


from pyspark.ml.pipeline import Pipeline

pipeline = Pipeline(
    stages=[assembler, scaler, lr]
)


# ### Task 7 - Split the data
# 

# In[43]:


# Split the data into training and testing sets with 70:30 split.
# set the value of seed to 42

(trainingData, testingData) = df.randomSplit((.7, .3), seed=42)



# ### Task 8 - Fit the pipeline
# 

# In[44]:


# Fit the pipeline using the training data

pipelineModel = pipeline.fit(trainingData)


# #### Part 2 - Evaluation
# 

# 
# **Run the code cell below.**<br>
# **Use the answers here to answer the final evaluation quiz in the next section.**<br>
# **If the code throws up any errors, go back and review the code you have written.**</b>
# 

# In[45]:


print("Part 2 - Evaluation")
print("Total rows = ", rowcount4)
ps = [str(x).split("_")[0] for x in pipeline.getStages()]

print("Pipeline Stage 1 = ", ps[0])
print("Pipeline Stage 2 = ", ps[1])
print("Pipeline Stage 3 = ", ps[2])

print("Label column = ", lr.getLabelCol())


# ## Part 3 - Evaluate the Model
# 

# ### Task 1 - Predict using the model
# 

# In[46]:


# Make predictions on testing data
predictions = pipelineModel.transform(testingData)


# ### Task 2 - Print the MSE
# 

# In[48]:


from pyspark.ml.evaluation import RegressionEvaluator

mse_evaluator = RegressionEvaluator(
    predictionCol='prediction',
    labelCol='SoundLevelDecibels',
    metricName='rmse'
)

mse = mse_evaluator.evaluate(predictions)
print(mse)


# ### Task 3 - Print the MAE
# 

# In[ ]:


#your code goes here

#TODO
mae = #TODO
print(mae)


# ### Task 4 - Print the R-Squared(R2)
# 

# In[ ]:


#your code goes here

#TODO
r2 = #TODO
print(r2)


# #### Part 3 - Evaluation
# 

# 
# **Run the code cell below.**<br>
# **Use the answers here to answer the final evaluation quiz in the next section.**<br>
# **If the code throws up any errors, go back and review the code you have written.**</b>
# 

# In[ ]:


print("Part 3 - Evaluation")

print("Mean Squared Error = ", round(mse,2))
print("Mean Absolute Error = ", round(mae,2))
print("R Squared = ", round(r2,2))

lrModel = pipelineModel.stages[-1]

print("Intercept = ", round(lrModel.intercept,2))


# ## Part 4 - Persist the Model
# 

# ### Task 1 - Save the model to the path "Final_Project"
# 

# In[ ]:


# Save the pipeline model as "Final_Project"
# your code goes here


# ### Task 2 - Load the model from the path "Final_Project"
# 

# In[ ]:


# Load the pipeline model you have created in the previous step
loadedPipelineModel = #TODO


# ### Task 3 - Make predictions using the loaded model on the testdata
# 

# In[ ]:


# Use the loaded pipeline model and make predictions using testingData
predictions = #TODO


# ### Task 4 - Show the predictions
# 

# In[ ]:


#show top 5 rows from the predections dataframe. Display only the label column and predictions
#your code goes here


# #### Part 4 - Evaluation
# 

# 
# 
# **Run the code cell below.**<br>
# **Use the answers here to answer the final evaluation quiz in the next section.**<br>
# **If the code throws up any errors, go back and review the code you have written.**</b>
# 

# In[ ]:


print("Part 4 - Evaluation")

loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[0].getInputCols()

print("Number of stages in the pipeline = ", totalstages)
for i,j in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {i} is {round(j,4)}")


# ### Stop Spark Session
# 

# In[ ]:


spark.stop()


# ## Authors
# 

# [Ramesh Sannareddy](https://www.linkedin.com/in/rsannareddy/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMBD0231ENSkillsNetwork866-2023-01-01)
# 

# ### Other Contributors
# 

# ## Change Log
# 

# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2023-05-26|0.1|Ramesh Sannareddy|Initial Version Created|
# 

# Copyright © 2023 IBM Corporation. All rights reserved.
# 