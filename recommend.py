import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv(r'C:\Users\srira\OneDrive\Desktop\project\ratings_Electronics.csv', header=None)

# Add column names
df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']

# Drop the 'timestamp' column
df = df.drop('timestamp', axis=1)

# Ensure 'rating' column is numeric, coercing errors to NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop rows with NaN values in 'rating' column
df = df.dropna(subset=['rating'])

# Copy the data to another dataframe
df_copy = df.copy(deep=True)

# Print number of rows and columns
rows, columns = df.shape
print("No of rows =", rows)
print("No of columns =", columns)

# Display DataFrame information
df.info()

# Find number of missing values in each column
print(df.isna().sum())

# Summary statistics of 'rating' variable
print(df['rating'].describe())

# Create and display a plot of the rating distribution
plt.figure(figsize=(12, 6))
df['rating'].value_counts(1).plot(kind='bar')
plt.show()

# Number of unique user ids and product ids in the data
print('Number of unique USERS in Raw data =', df['user_id'].nunique())
print('Number of unique ITEMS in Raw data =', df['prod_id'].nunique())

# Top 10 users based on rating
most_rated = df.groupby('user_id').size().sort_values(ascending=False)[:10]

# Filter users with at least 50 ratings
counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

print('The number of observations in the final data =', len(df_final))
print('Number of unique USERS in the final data =', df_final['user_id'].nunique())
print('Number of unique PRODUCTS in the final data =', df_final['prod_id'].nunique())

# Creating the interaction matrix
final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
print('Shape of final_ratings_matrix:', final_ratings_matrix.shape)

# Finding the number of non-zero entries in the interaction matrix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings =', given_num_of_ratings)

# Finding the possible number of ratings
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings =', possible_num_of_ratings)

# Density of ratings
density = (given_num_of_ratings / possible_num_of_ratings) * 100
print('density: {:4.2f}%'.format(density))

# Calculating the average and count of ratings for each product
average_rating = df_final.groupby('prod_id')['rating'].mean()
count_rating = df_final.groupby('prod_id')['rating'].count()

# Create a dataframe with calculated average and count of ratings
final_rating = pd.DataFrame({'avg_rating': average_rating, 'rating_count': count_rating})

# Sort the dataframe by average rating
final_rating = final_rating.sort_values(by='avg_rating', ascending=False)
print(final_rating.head())

# Function to get top n products based on highest average rating and minimum interactions
def top_n_products(final_rating, n, min_interaction):
    recommendations = final_rating[final_rating['rating_count'] > min_interaction]
    recommendations = recommendations.sort_values('avg_rating', ascending=False)
    return recommendations.index[:n]

print(list(top_n_products(final_rating, 5, 50)))
print(list(top_n_products(final_rating, 5, 100)))

# Interaction matrix with 'user_index'
final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])
final_ratings_matrix.set_index(['user_index'], inplace=True)

# Function to get similar users based on cosine similarity
def similar_users(user_index, interactions_matrix):
    similarity = []
    for user in range(0, interactions_matrix.shape[0]):
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))
    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity if tup[0] != user_index]
    similarity_score = [tup[1] for tup in similarity if tup[0] != user_index]
    return most_similar_users, similarity_score

print(similar_users(3, final_ratings_matrix)[0][0:10])
print(similar_users(3, final_ratings_matrix)[1][0:10])
print(similar_users(1521, final_ratings_matrix)[0][0:10])
print(similar_users(1521, final_ratings_matrix)[1][0:10])

# Function to get recommendations based on similar users' preferences
def recommendations(user_index, num_of_products, interactions_matrix):
    most_similar_users = similar_users(user_index, interactions_matrix)[0]
    prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)])
    recommendations = []
    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            similar_user_prod_ids = set(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)])
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break
    return recommendations[:num_of_products]

print(recommendations(3, 5, final_ratings_matrix))
print(recommendations(1521, 5, final_ratings_matrix))

# Convert the interaction matrix to sparse matrix format
from scipy.sparse import csr_matrix
final_ratings_sparse = csr_matrix(final_ratings_matrix.values)

# Perform Singular Value Decomposition
U, s, Vt = svds(final_ratings_sparse, k=50)  # k is the number of latent features
sigma = np.diag(s)

print(U.shape)
print(sigma.shape)
print(Vt.shape)

# Compute the predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Convert to DataFrame
preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=final_ratings_matrix.columns)
preds_matrix = csr_matrix(preds_df.values)

# Function to recommend items based on predicted ratings
def recommend_items(user_index, interactions_matrix, preds_matrix, num_recommendations):
    user_ratings = interactions_matrix[user_index,:].toarray().reshape(-1)
    
    
    user_predictions = preds_matrix[user_index,:].toarray().reshape(-1)
    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Products'] = np.arange(len(user_ratings))
    temp = temp.set_index('Recommended Products')
    temp = temp.loc[temp.user_ratings == 0]
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended products for user(user_id = {}):\n'.format(user_index))
    print(temp['user_predictions'].head(num_recommendations))

# Generate recommendations
recommend_items(121, final_ratings_sparse, preds_matrix, 5)
recommend_items(100, final_ratings_sparse, preds_matrix, 10)

# Calculate RMSE for SVD model
final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])
final_ratings_matrix.set_index(['user_index'], inplace=True)

average_rating = final_ratings_matrix.mean()
avg_preds = preds_df.mean()

rmse_df = pd.concat([average_rating, avg_preds], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']

RMSE = mean_squared_error(rmse_df['Avg_actual_ratings'], rmse_df['Avg_predicted_ratings'], squared=False)
print(f'RMSE SVD Model = {RMSE} \n')
