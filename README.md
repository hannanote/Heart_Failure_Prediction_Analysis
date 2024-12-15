### The Use of Machine Learning Algorithms for Cardiovascular Disease Analysis and Prediction 

Cardiovascular diseases (CVDs) are a group of disorders of the heart and blood vessels. Heart attacks are acute events mainly caused by a blockage that prevents blood from circulating to the heart. According to the World Health Organization (WHO), out of the 17 million premature deaths (under the age of 70) due to noncommunicable diseases in 2019, 38% were caused by CVDs. (World Health Organization, Cardiovascular diseases (CVDs), 2021, June 11). Also, approximately 31% of global deaths are attributed to heart failure (Shrivastava et al., 2015).  

Data analysis with categorization is essential for heart disease prediction. The severity of the disease can be categorized using machine learning (ML) tools such as the K-Nearest Neighbor (KNN) algorithm, Decision Tree (DT), and Naïve Bayes (NB) algorithm. (M.Gandhi, 2015). Machine learning (ML) is a branch of artificial intelligence (AI) that uses data and algorithms to imitate how humans learn, gradually improving accuracy.  

#### Machine Learning Algorithms 

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


![image](https://github.com/user-attachments/assets/85195635-841c-425b-87df-529857cd3e51)

![image](https://github.com/user-attachments/assets/ab3e9b80-a2da-4119-9484-c50d7c460432)

![image](https://github.com/user-attachments/assets/824fd774-1634-4d18-87a1-a16b0b83f197)

![image](https://github.com/user-attachments/assets/a67f95ad-b3ef-49ed-bb85-b1a239e4f26e)

![image](https://github.com/user-attachments/assets/20ba1b69-2500-432f-a0f2-2be877c0d982)


#### References

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data

