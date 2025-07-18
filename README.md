Netflix Customer Churn Prediction 
In the highly competitive landscape of subscription-based streaming services, customer 
retention is crucial for sustained business growth. Customer churn, the phenomenon 
where users cancel their subscriptions, directly impacts revenue, profitability, and 
market share. With the increasing presence of alternative streaming platforms and 
shifting viewer preferences, it is essential for companies like Netflix to understand why 
customers leave and proactively address churn risk. 
Derived from that problem, we developed a machine learning model to predict 
customer churn based on various factors such as user engagement, payment history, 
and demographic information. This project utilizes the Netflix Engagement Dataset 
sourced from Kaggle to analyze and classify customer churn patterns. By identifying 
high-risk customers early, Netflix or other subscription-based platforms can implement 
targeted retention strategies, including personalized recommendations, exclusive 
content, special discounts, or improved customer support. 
For this study, churn labels are defined as follows: 
0 = No Churn (Customer remains subscribed) 
1 = Churn (Customer cancels subscription) 
To evaluate the model's effectiveness, we prioritize recall as the primary performance 
metric. This is to minimize false negatives, ensuring that customers who are actually at 
risk of churning are correctly identified. By emphasizing recall, we reduce the chances of 
misclassifying a high-risk customer as a retained one, allowing businesses to take 
proactive retention measures before they cancel their subscription. 
2.Tools & Technologies Used 
• Programming Language 
• Python was the core programming language used for all data analysis, visualization, and 
machine learning implementation. 
• ● Libraries and Frameworks 
• Pandas: For data manipulation and preprocessing (cleaning, transformation, grouping). 
• Matplotlib & Seaborn: For static visualizations such as pie charts, step plots, heatmaps, 
and boxplots. 
3 
• Scikit-learn: For building classification models, data splitting, scaling, encoding, 
evaluation, and metrics. 
• NumPy: For handling numerical operations and matrix transformations. 
• PCA from sklearn.decomposition: Used for optional dimensionality reduction. 
•  Environment 
• The entire project was executed in a Jupyter Notebook environment, allowing for step
by-step code execution, inline visualization, and markdown-based documentation. 
• POWER BI  ----- Loading, inspecting, and preparing customer churn data for analysis 
3. Areas Covered During Training 
During the course of this project and training, the following areas were explored and 
implemented: 
• Data Collection and Cleaning 
→ Loading, inspecting, and preparing customer churn data for analysis 
• Feature Selection and Encoding 
→ Selecting relevant features for churn prediction and encoding categorical variables 
• Supervised Machine Learning Concepts 
→ Understanding classification problems and applying Logistic Regression 
• Model Training and Evaluation 
→ Splitting data into training and testing sets 
→ Fitting a classification model and evaluating it using accuracy, precision, recall, F1
score, and classification reports 
1. Handling Imbalanced Datasets 
→ Using class_weight='balanced' to handle uneven churn and non-churn cases 
• Data Visualization and Interpretation 
→ Creating line plots and count plots to compare actual vs predicted churn outcomes 
• Generating Project Reports and Insights 
→ Summarizing results, preparing documentation, and presenting key findings for 
business decision-making 
4.Implementation 
4 
First, we import all necessary libraries that we need to develop for the machine learning 
models in this project. 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, classification_report 
import seaborn as sns 
import matplotlib.pyplot as plt 
After importing all the necessary libraries, the next step is to load the dataset and analyze it to 
gain a better understanding of its structure and contents 
df = pd.read_csv("F:\\INT375\\Netflix Engagement Dataset.csv") 
The data has 3500 entries and 16 features including 1 target Churn Status column. 
Next is to explore more about the dataset structure. 
print("Dataset Shape:", df.shape) 
df.info() 
5 
Key Observations: 
• The Subscription Length varies from 1 to 60 months, with an average of 30.5 months. 
• Customer Satisfaction Score ranges from 1 to 10, with a mean of 6.93. 
6 
• Daily Watch Time is between 1 to 6 hours, averaging 3.5 hours. 
• The dataset includes Age (18-70 years) and Monthly Income (1,010−9,990). 
• The Subscription Length varies from 1 to 60 months, with an average of 30.5 months. 
• Customer Satisfaction Score ranges from 1 to 10, with a mean of 6.93. 
• Daily Watch Time is between 1 to 6 hours, averaging 3.5 hours. 
• The dataset includes Age (18-70 years) and Monthly Income (1,010−9,990).
