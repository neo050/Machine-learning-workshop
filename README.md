# Machine-learning-workshop-step-A
I chose at least 4 different tools to perform features extraction and features selection and used them on a database of my choice.


"

Requirements document for phase one of the project in the machine learning workshop
Features extractions
This phase of the project will be carried out by individuals only.
This phase of the project is designed to give you knowledge about existing tools for performing feature extractions. For this purpose, it is necessary to select an existing database with information in order to experiment with the tools on it. Examples of databases are listed at the end of this document. There is no obligation to choose one of them.


You must choose at least 4 different tools to perform features extraction and features selection and use them on a database of your choice.
You are required to submit a short overview (no more than a page) detailing all the steps you performed with each of the tools on your database.
During the tests, make sure to discover and experiment with the different capabilities of each tool.
In the final report, in addition to detailing all the steps taken, several paragraphs summarizing the ease and comfort of working with each tool as well as comparisons between the different tools - which was more convenient? What less? Which tool is better for one type of data and which for another?
Try to think ahead to your possible future uses of each of the tools and rate which one will be more useful to you and when. Think about what you learned from this experience and describe it.
It is mandatory to attach a bibliographic list to the work (scientific sources from which you obtained the information about the specific tools / setting)!

Please note that you are welcome to use any available, reliable and large enough database to carry out the project. If you found an independent database, you must include the source of the information in the review itself.
Possible sources of information for projects:
• https://www.kaggle.com/datasets - over 100 databases uploaded by surfers (there are several esoteric and interesting topics such as statistics on computer games)
• https://www.data.gov/ - databases uploaded by the US government, there is a lot of information about social issues here
• https://grouplens.org/datasets/movielens/ Huge database of movies and movie ratings.

"
# Step A is done
"
Presenter: Nahorai Hagag
ID : *****
GIT
DATABASE

Project Overview: Feature Extraction and Selection on Drug Overdose Death Rate Data

introduction:
This project aims to investigate and apply different tools for Feature Extraction and Feature Selection on a dataset detailing drug overdose death rates in the United States, segmented by drug type, sex, age, race and Hispanic origin. The goal is to identify the most significant features that influence drug overdose mortality rates and to understand the underlying structure of the data set using dimensionality reduction techniques.

Tools I used:
Pandas-used for initial manipulation of the data, including extracting new columns.
2. Feature-engine's RareLabelEncoder- used to handle rare labels in the 'ethnicity' category.
3. Pandas' get_dummies - used for coding categorical variables ('gender' and 'ethnicity').
4. SelectKBest with f_classif- used for feature selection based on statistical tests.
5. StandardScaler- used to expand features before running PCA.
6. PCA (Principal Component Analysis) - used to reduce dimensions.
   
methodology:
Data Preprocessing- Initialized by loading the dataset using Pandas and extracting 'Gender' and 'Ethnicity' columns from the 'STUB_LABEL' column. I combined rare labels within 'ethnicity' using Feature-engine's RareLabelEncoder, then encoded 'gender' and 'ethnicity' with Pandas' get_dummies function.
Dimensionality Reduction- I implemented SelectKBest with f_classif to identify the top 10 features that influence the 'estimation' of drug overdose death rates.
Dimensionality Reduction- PCA was performed after rescaling to reduce the data set to two principal components for visualization and further analysis.

Analysis and comparison of tools:
Pandas have proven indispensable for basic data manipulation, offering intuitive syntax and powerful capabilities for handling DataFrame operations. The ease of extracting and creating new columns was exceptional, making it an essential tool for data preprocessing.
Feature-engine's RareLabelEncoder was especially useful for managing categorical variables with many levels. This simplified the dataset and improved the effectiveness of subsequent analysis by grouping rare categories, improving model interpretation.
  -Pandas' get_dummies function was simple for coding categorical variables, but it lacked the flexibility and advanced options provided by dedicated coding libraries.
SelectKBest with f_classif - proposed a simple but effective method for feature selection, identifying significant variables efficiently. However, its usefulness may be limited in more complex arrays where feature interactions are essential.
StandardScaler- was critical to preparing the data for PCA, ensuring that each feature contributed equally to the analysis. Its simplicity and efficiency make it a recommended choice for feature scaling.
PCA allowed a deep understanding of the structure of the data set, highlighting the variance explained by the principal components. Despite its power, its abstract nature requires careful interpretation to link back to the original features and their meanings.

Reflections and future use:
Each tool has its own strengths and is suitable for different aspects of the data analysis process. Pandas is second to none for data manipulation, while Feature-engine and SelectKBest excel at feature engineering and selection, respectively. PCA offers deep insights into the structure of the data, making it invaluable for complex datasets.

In future projects, the choice of tool will depend on the specific requirements of the task at hand. For example, Pandas will remain a fundamental component of data manipulation, while SelectKBest and PCA will be essential for projects involving feature selection and dimensionality reduction, especially in large datasets.

bibliography
McKinney, W. (2012). Python for Data Analysis. O'Reilly Media, Inc.
Feature-engine documentation. https://feature-engine.readthedocs.io
Scikit-learn documentation. https://scikit-learn.org/stable/
"
