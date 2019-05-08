# Santander Product Recommendation using Collaborative Filtering
Customer analytics project using the Santander CRM data to create product recommendations

## 1	Abstract

Santander is a multinational bank in Spain offering an array of financial services and products. Under their current system, a small number of Santander’s customers receive many recommendations while many others rarely see any resulting in an uneven customer experience. Models were built to aid the bank in selling products through mass marketing as well as personalized recommendations.
Customers were segmented based on the number of relations they held with the bank. All customers having more than two accounts were considered and equal width segmentation was applied. Market basket analysis was performed in all three segments and the association rules derived from each segment were used to select the potential products for mass marketing strategies. Those products in rules which had a high lift and low expected confidence were identified as high potential low selling products which could be mass marketed. 
Santander’s core business involves a client relationship manager guiding a client’s investment decisions.  Recommender systems can aid relationship managers with making personalized and automated selection of next best products for private banking clients. Different types of collaborative filtering recommender systems were built to help select the next best offer. Since the data had information only about the presence or absence of a product with a customer, Implicit Feedback CF with Alternating Least Squares was implemented where products were modelled as a function of Preference and Confidence. This system was improved by a user-user similarity-based system and further enhanced by incorporating demographic correlations. For the enhanced model, pairwise user-user similarity using cosine similarity using user-demography vectors was calculated. For each user, based on the enhanced similarity scores, 1000 users were identified using the k-Nearest Neighbors technique. The items to be recommended were predicted as a weighted average of the preferences from each user’s neighborhood. Precision and Recall @ K were used to evaluate the recommender systems.  
Index terms – Equal width segmentation, market basket analysis, association rules, recommender systems, implicit feedback collaborative filtering, alternating least squares, demographic correlations.

## 2	Introduction

Banco Santander, S.A., doing business as Santander Group, is a Spanish multinational commercial bank and financial services company founded and based in Santander, Spain. It offers an array of financial services and products including retail banking, mortgages, corporate banking, cash management, credit card, capital markets, trust and wealth management, and insurance. The bank is interested in improving its personal product recommendation system for its customers. It collects marketing information about its customers and associates it with bank account products. Under their current system, a small number of Santander’s customers receive many recommendations while many others rarely see any resulting in an uneven customer experience. With an effective recommendation system in place, Santander can better meet the individual needs of all customers and ensure their satisfaction.

## 3	Objective
In this project, we will analyze the Santander’s bank customer profiles, data, products and transactions. Our aim is to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers.  The analysis will include the following:  
a.	Identify different approaches of segmentation based on customer’s needs or values to market products and select the best approach
b.	Perform Market Basket Analysis for the segments in the best segmentation approach and market the rules through different techniques.
c.	Build recommendation engine to engage in personalized marketing.

## 4	Analysis Process Flow

## 5 Collaborative Filtering for Product Recommendation

### 5.1	Collaborative Filtering for Implicit Feedback datasets

A common task of recommender systems is to improve customer experience through personalized recommendations based on prior implicit feedback. These systems passively track different sorts of user behavior, such as purchase history, watching habits and browsing activity, in order to model user preferences. For the implicit feedback in product ownership, we will use the matrix factorization methodology using ALS to perform the recommendations. To perform this, we start with a USER-ITEM matrix R of size u x i with our users, items, which is required to be decomposed into matrices with users and hidden features of size u x f and one with items and hidden features of size f x i. In U and V we have weights for how each user/item relates to each feature as below:

 
Figure 29: Latent Factorization of User-Item matrix

The implicit feedback is modelled as function of PREFERENCE and CONFIDENCE on the PREFERENCE that a user likes a product based on his purchase. We start with missing values, which imply user-account cells of the accounts that a user has never interacted with as a negative preference with a low confidence value and existing values which are the accounts a user owns a positive preference but with a high confidence value. The preference P, is set as a binary representation of our feedback data r. If the feedback is greater than zero we set it to 1:
 
As a next step, we calculate the confidence (c) on the preference (p) as below using the magnitude of r:
 
The rate of which our confidence increases is set through a linear scaling factor α. We also add 1 so we have a minimal confidence even if α x r equals zero. Now the next step is to find the vector for each user (xu) and account (yi) in latent factor dimensions in a way that it minimizes the loss function:

 
Where the first term is the sum squares of the difference between the actual interaction between a user and an account and the calculated rating as a product of the preference and confidence from the user and item factor matrices. The second term is the squared magnitude of the factor vectors multiplied by the regularization parameter to maintain the bias-variance balance and prevent overfitting. The presence of the X transpose time Y matrix term renders this function non-convex, so conventional optimization methods like Gradient descent will take a large amount of memory for fitting on the 0.68M x 24 sized matrix, so as suggested by the research paper, if we fix the user factors or item factors we can calculate a global minimum as one of terms being constant makes the objective function convex. The derivative of the above equation produces the following equation for minimizing the loss to produce the user factors is:

 

The loss function to be reduced to produce the factors for the accounts is:

 

By iterating between computing the two equations above we create one matrix with user vectors and one with item vectors that we can then use to produce recommendations or find similarities. To make recommendations for a given user we calculate the dot product between our user vector and the transpose of our item vectors. This gives us a recommendation score for our user and each item which is later scaled to a 0 to 1 scale for sanity:
 
We can iterate between users and items to identify the scores of every item for each user and rank them by order of preference and confidence. Then the top items are used to make recommendations to the user. 
The training of the model using the IMPLICIT library’s fit function, is done with 50 latent factors, the regularization factor which is the inverse of 𝛌 is chosen as 0.1, and the alpha value is taken to 40 as suggested in the paper.

 
Figure 30: Collaborative Filtering for Implicit Feedback using ALS

From the produced recommendations, it can be seen that the model recommends most, some of the very popular products among the customers like the particular account, direct debits account and the e-account.

 
Figure 31: Top recommended products
Performing model validation by applying the top recommendation to each user for a sample of 10000 users and scoring the results against the last month data of the actual purchases by the customers produces below results:

 
Figure 32: Collaborative Filtering (ALS) - Model Results

The precision and recall of the model are considerably low, with it only being able to capture 2328 of the 10000 purchases. We will also build other recommender system models and compare them on a similar scale by their performance on the score data.

### 5.2	User-User similarity-based CF
User based Collaborative filtering uses cosine similarity to find similar users to the target user and compares the products which both users own and recommends the product which target user doesn’t have and will most likely buy due to the similarity in purchase behavior between the users.

 
Figure 33: Example of USER-USER based CF
For Santander dataset, A data-item matrix is created for the one but last month data. Then, USER-USER cosine similarity matrix for each user pair is constructed. The system is tested for a single user first and then expanded to the entire dataset. For a single user, 500 most similar users and their products are identified. Considering the products the user already owns, the most similar items to the ones the user has liked from the neighborhood are identified. The top 5 most similar items are recommended to the user sorted by score. The below image shows the example for a selected user: 15889 who owns current account, particular plus and securities and the top 5 recommendations for the user are shown too.

 
Figure 34: Top 5 recommendations for user 15889
This process is performed for all the users in dataset and the most recommended products are found out.
From the produced recommendations, the model recommends most, some of the very popular products among the customers like the particular account, direct debits account and the e-account.

 
Figure 35: Top recommended products
Performing model validation by applying the top recommendation to each user for a sample of 10000 users and scoring the results against the last month data of the actual purchases by the customers produces below results:

 
Figure 36: CF for Binary Implicit Feedback - Model Results
The precision and recall of the model are 30% more than CF-ALS model, with it being able to capture 3149 of the 10000 purchases. We will enhance this recommender system by incorporating demographic correlations.

### 5.3	User-User similarity-based CF enhanced by Demographic Correlations
Instead of focusing on users and products alone, we decided to add demographic features in the existing user-based CF model by using a hybrid algorithm that keeps the core idea of the existing User-based CF recommender system and enhances them with relevant information extracted from demographic data.
The following key demographic attributes were considered and one-hot encoded: sex, age bin, new customer index, seniority bin, foreign index, province name. User profiles were expressed as vectors constructed solely from demographic data and similarities among those user profiles were calculated for final predictions to be generated. 

 
Figure 37: User-Product & Demographics Matrix
Demographic correlations between two users are defined by the similarity of the vectors which represent the specific users. That similarity is calculated by the cosine similarity of the two vectors. In the above image, the first matrix shows the calculation of similarities between users based on products and the second matrix shows similarities between users based only on demographics. 
Using the above 2 user-user similarity matrices, enhanced similarity matrix is obtained using the below formula where UU Sim user_item = User – Product matrix, UU Sim user_demography   = User- Demography matrix
  
Figure 38: User-based CF enhanced by demographic correlations
•	After the enhanced matrix is generated, 1000 nearest neighbors were identified.

•	Items to be recommended were predicted as the weighted average of the preferences from each user’s neighborhood.

•	Items already owned by user were removed and the remaining recommendations were ranked by score.

From the produced recommendations, the model recommends most, some of the very popular products among the customers like the direct debits account, particular account and the taxes account.
 
Figure 39: Top recommended products
Performing model validation by applying the top recommendation to each user for a sample of 10000 users and scoring the results against the last month data of the actual purchases by the customers produces below results:

 
Figure 40: CF with demographic correlations - Model Results
The precision and recall of the model are the highest among all models, with it being able to capture 3480 of the 10000 purchases. There is a slight improvement in the metrics in this enhanced model over the base User-User model.

### 5.4	Model comparison
We have built three Collaborative filtering-based recommender systems of which two are enhanced and one baseline model to compare the uplift. Now we will perform model comparison based on their performance on the score data to select one for implementation in Santander production. To compare the models, we will use @K metrics because in the context of recommendation systems we are interested in recommending top-N items to the user and expect the recommendation to turn into conversion. So, the evaluation is valid to compute precision and recall metrics in the first N items instead of all the items. Thus, we’ve tended to precision and recall at k, where k is a user definable integer to match the top-N recommendations objective:

•	Precision @ K: Precision at k is the proportion of recommended items in the top-k set that are relevant i.e. that a user ends up purchasing. It’s interpreted as, if precision at 5 in a top-5 recommendation problem is 65%. This means that 65% of the recommendations made are relevant to the user.

Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)

•	Recall @ K: Recall at k is the proportion of relevant items found in the top-k recommendations. It’s interpreted as, if recall at 5 is found to be 45% in our top-5 recommendation system, it means that 45% of the total number of the relevant items are captured in the top-k recommended items.

Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)

The recommendation is done on a score cut off basis, so if the top recommended account falls below the cut off, there is no recommendation offer made. So, if there are no items recommended. i.e. number of recommended accounts at k is zero, we cannot compute precision at k since we cannot divide by zero. In that case we set precision at k to 1. This makes sense because in that case we do not have any recommended account that is not relevant. Similarly, when computing Recall@k, when the total number of relevant items is zero, i.e. if a customer never purchased a new product in the next month, we set recall at k to be 1. This is because we do not have any relevant/purchased account that is not identified in our top-k results. So, the final Precision and Recall is calculated as the average of individual Precision and Recall metrics per customer.

Owing the business scenario in a bank, there is a high chance of customer fatigue if the number of offers/number of times a relationship manager reaches out to a customer with an offer to take up a new engagement is high, causing customer satisfaction indexes to fall or worst case, customer churn, so we set K=3, as the maximum applicable for the use-case, but iterate from K=1 to K=5 to compare the model performance.  

PRECISION at K metrics for each model:
	K=1	K=2	K=3	K=4	K=5
IMPLICIT FB LATENT FACTORIZATION BY ALS	0.233	0.2238	0.183	0.15	0.143
USER BASED CF FOR BINARY IMPLICIT FB	0.315	0.28	0.232	0.209	0.188
USER BASED CF WITH DEMOGRAPHIC CORRELATION	0.344	0.289	0.242	0.21	0.189


RECALL at K metrics for each model:
	K=1	K=2	K=3	K=4	K=5
IMPLICIT FB LATENT FACTORIZATION BY ALS	0.22	0.44	0.51	0.55	0.66
USER BASED CF FOR BINARY IMPLICIT FB	0.29	0.53	0.64	0.77	0.866
USER BASED CF WITH DEMOGRAPHIC CORRELATION	0.32	0.54	0.67	0.78	0.87

Before choosing a final model, we can see that the Latent Factorization model optimized using ALS performs poorest on the Santander data. This could be because of the implicit feedback being only unary which makes the modelling of the product ownership data into preference and confidence no different from using the raw data in itself as both Ru-i and Cu-I terms are into [0,1] binary scale. Also, it’s evident that the introduction of the demographic correlations improves on the baseline USER-USER similarity collaborative filtering model, so there exists an effect between factors like the user’s personal profile and location of residence with the preference of banking products purchased by them. So the user similarity based collaborative filtering enhanced by demographic correlations recommender system model is chosen as the final model as it’s able to capture almost 70% of the relevant accounts for the users within the allowable limit of K=3. To put this into perspective, there are 0.68M customers in the Santander bank’s database and only 23K of them ended up purchasing an additional account in the score data month, so a recall of 70% is considerably good for production implementation.

Choice of an ideal K represents these metrics as inputs for a typical elbow graph problem to identify the trade- off. We can see the elbow is sharpest at K=2, however we can choose the ideal K=3 for production implementation of the recommender engine as in the bank’s scenario the cost of the false positive (which indicates a recommendation not ending up in a purchase) is much lower than a false negative (which indicates the failure to predict an actual made purchase).

 
Figure 41:  Elbow graph for PRECISION and RECALL @ K

