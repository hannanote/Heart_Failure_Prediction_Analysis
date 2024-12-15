## The Use of Machine Learning Algorithms for Cardiovascular Disease Analysis and Prediction 

Cardiovascular diseases (CVDs) are a group of disorders of the heart and blood vessels. Heart attacks are acute events mainly caused by a blockage that prevents blood from circulating to the heart. According to the World Health Organization (WHO), out of the 17 million premature deaths (under the age of 70) due to noncommunicable diseases in 2019, 38% were caused by CVDs. (World Health Organization, Cardiovascular diseases (CVDs), 2021, June 11). Also, approximately 31% of global deaths are attributed to heart failure (Shrivastava et al., 2015).  

Data analysis with categorization is essential for heart disease prediction. The severity of the disease can be categorized using machine learning (ML) tools such as the K-Nearest Neighbor (KNN) algorithm, Decision Tree (DT), and Naïve Bayes (NB) algorithm. (M.Gandhi, 2015). Machine learning (ML) is a branch of artificial intelligence (AI) that uses data and algorithms to imitate how humans learn, gradually improving accuracy.  

This project involves predicting cardiovascular disease (CVD) using machine learning techniques. A variety of models were trained and evaluated to identify the best-performing model for predicting the likelihood of heart disease based on patient data. The goal was to develop a reliable classifier that can assist in early diagnosis and decision-making.

### Dataset and Features

The dataset is obtained from the Kaggle heart disease dataset. This dataset contains 11 features that can be employed to anticipate the likelihood of heart disease: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting electrocardiogram results, maximum heart rate, exercise-induced angina, the slope of the peak exercise ST segment, and heart disease.

![image](https://github.com/user-attachments/assets/85195635-841c-425b-87df-529857cd3e51)
Description of Features


### EDA

![image](https://github.com/user-attachments/assets/ab3e9b80-a2da-4119-9484-c50d7c460432)


### Machine Learning Algorithms 

Machine learning involves the exploration and modeling of extensive datasets. It organizes data into large sets to discover patterns and analyze their relationships. Its popularity in the medical field stems from its capability to handle extensive, intricate, and diverse data, with prediction being one of its notable applications (Pouriyeh et al., 2017). Several machine learning techniques have been utilized for predicting issues related to heart disease. Figure 1 shows some examples of the heart disease prediction data used in this research.   

##### Logistic Regression 

Logistic regression (LR) is a statistical approach often used to solve binary classification issues. Rather than fitting a straight line, logistic regression employs the logistic function to constrain the output of a linear equation to the range of 0 to 1. Due to 11 independent variables and one outcome variable, logistic regression is well-suited for categorization.

##### Random Forest 

Classification and regression techniques based on Random Forest (RF) are also utilized in our study. Significant improvements in classification accuracy have resulted from growing an ensemble of trees and letting them vote for the most popular class. To grow these ensembles, random vectors are generated that govern the growth of each tree in the ensemble. In classification, random forest is a classifier consisting of a collection of trees-structured classifiers {h(x,k ), k = 1, . . .} where the {k} are independent identically distributed random vectors and each tree cast a unit vote for the most popular class at input x (LEO BREIMAN, 2001) 

##### Decision Tree 

Decision Tree (DT) learns a sequence of “if-then” questions with each question involving one feature and one split point. The tree carves up the feature space into groups of observations that share similar target values, and each leaf represents one of these groups. Each leaf in the decision tree is responsible for making a specific prediction. For classifier trees, the prediction is a target category such as heart disease or not at our study. 

##### K-Nearest Neighbors 

In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is assigned to the class of that single nearest neighbor (WiKi, 2023).  K-NN has some strong consistency results. As the amount of data approaches infinity, the two-class k-NN algorithm is guaranteed to yield an error rate no worse than twice the Bayes error rate (the minimum achievable error rate given the data distribution) (Thomas M.; Hart, Peter E., 1967).    

 
##### Support Vector Machine 

Support vector machines (SVM) have a solid ability to approximate linear and nonlinear relationships. SVMs work by mapping the input vectors into some high-dimensional feature space using nonlinear functions. An essential idea behind SVMs is that not all the data points influence optimality as in linear regression or Naive Bayes, but the points close to the decision boundary. These points are called Support Vectors, hence the name Support Vector Machine.  

 
##### Gradient Boosting  

Gradient Boosting is an ensemble learning technique that builds a strong predictive model by combining the predictions of multiple weak learners, usually decision trees. Gradient Boosting works sequentially, where each new tree corrects the errors of the previous one. It pays more attention to the instances that were misclassified by the earlier trees. 

##### AdaBoost  

AdaBoost (Adaptive Boosting) is another ensemble learning model. It focuses on improving the performance of weak learners to create a strong model. Like Gradient Boosting, AdaBoost builds weak learners sequentially. Each new learner corrects the mistakes of the previous ones. AdaBoost can work with various weak learners, and it is not limited to decision trees. As a result, the AdaBoost increases the weight of misclassified observations and decrease the weight of correctly identified observations. In the following iterations, higher emphasis is placed on misclassified observations. Ultimately, a stronger classifier is created by combining all the developed weak classifiers using a linear combination method, aiming for accurate classification performance (Ali, et al., 2021).


### Machine Learning Models 

Best Model: Random Forest Classifier
The Random Forest classifier demonstrated the best overall performance with:

Accuracy: 97.6%
Precision: 97%
Recall (Sensitivity): 98.6%
Specificity: 96.3%
F1 Score: 97.8%
Other Notable Performances:
The Gradient Boosting and Decision Tree classifiers performed well in terms of precision, but they were excluded from the final model due to dataset imbalances that could have led to misleading results. The SVM showed a solid precision rate of 84%, but its performance was not as strong across all evaluation metrics.





![image](https://github.com/user-attachments/assets/824fd774-1634-4d18-87a1-a16b0b83f197)

Model Evaluation Metrics:

The Random Forest achieved a high F1 score of 84.3%, balancing precision and recall effectively.
The model also performed well in Cross-validation (90.85%) and ROC-AUC (83.24%), indicating strong generalization capabilities and good ability to distinguish between positive and negative cases.



![image](https://github.com/user-attachments/assets/a67f95ad-b3ef-49ed-bb85-b1a239e4f26e)

Using the Random Forest model, the following top features were identified as most important for predicting heart disease:

+ ST_Slopes
+ Resting ECG
+ Chest Pain Types
+ Oldpeak
+ Exercise Angina

These features are particularly relevant in cardiology, where ECG results and chest pain types are critical for diagnosing cardiovascular conditions. Research has shown that ECG abnormalities, especially in older adults, are strong indicators of heart disease risk.

![image](https://github.com/user-attachments/assets/20ba1b69-2500-432f-a0f2-2be877c0d982)


### Conclusion 

The Random Forest Classifier has proven to be the most effective model for predicting cardiovascular disease (CVD) in this dataset, surpassing other machine learning models in key evaluation metrics such as accuracy, precision, recall, specificity, and F1 score. Key features like ECG readings, chest pain types, and exercise angina were identified as highly influential in predicting heart disease, with significant clinical relevance. These results demonstrate the potential of the Random Forest model to assist in early heart disease detection. The model provides a solid foundation for further research and could be seamlessly integrated into clinical decision support systems, improving heart disease diagnosis and patient outcomes.

### Limitations and Future Works

##### Feature Selection:

While we identified key features like ST_Slopes, Resting ECG, and Chest Pain Types, the current feature set is based solely on the dataset available. Important clinical factors such as lifestyle, genetic predisposition, and other health markers might be excluded. Incorporating additional, more comprehensive features could improve model performance and prediction accuracy.

##### Lack of Temporal or Longitudinal Data:

The dataset does not incorporate temporal information (e.g., changes in a patient’s health over time, history of heart disease events), which is often important in medical predictions. Heart disease risk evolves over time and considering how a patient’s health changes could lead to more accurate predictions. Without this temporal context, the model’s predictions may not fully reflect the dynamic nature of heart disease.

##### Incorporating Additional Features:

Future work should focus on gathering and integrating more comprehensive data, such as patient lifestyle factors (e.g., smoking, alcohol consumption, physical activity), genetic data, and biomarkers (e.g., cholesterol, blood pressure, glucose levels). These additional features could enhance the model's predictive power and make it more clinically relevant.

##### Model Optimization and Hyperparameter Tuning:

Further hyperparameter optimization through techniques like grid search or randomized search could refine the performance of the Random Forest model. Additionally, experimenting with other machine learning models such as XGBoost or LightGBM may lead to further improvements.


### References

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

