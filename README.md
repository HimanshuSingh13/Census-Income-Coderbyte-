# Census-Income-Coderbyte-
Census Income Prediction

The project aims to employ several supervised techniques to accurately predict individuals' income. The importance of this project lies in, for example, helping non-profit organizations evaluate their much-needed donation requests from different individuals.



Dataset

The dataset that will be used is the Census income dataset, which was extracted from the machine learning repository (UCI), which contains about 32561 rows and 15 columns. The target variable in the data set is income level, which shows whether a person earns more than 50,000 per year or not, based on 14 features containing information on age, education, education-num, gender, native-country, marital status, final weight, occupation, work classification, gender, race, hours-per-week, capital loss, and capital gain.

So, the target variable (income) will be represented by binary classes. the class 0 for people having income less than or equal to 50k $ per year (<=50k $), and the class 1 for people having income more than 50k $ per year (>50k $).



Requirements

This project requires Python 3.x and the following Python libraries installed:



NumPy

Pandas

matplotlib

Scikit-learn



Problem Statement:

This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) & (AGI>100) & (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year or less then.

Here’s a step-by-step outline of the project:



Importing necessary libraries.

Importing dataset from GitHub.

Exploratory Data Analysis (EDA).

Data Preprocessing & Feature Engineering.

Model building and Saving.
