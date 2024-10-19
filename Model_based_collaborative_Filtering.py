import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Import the data set
df = pd.read_csv(r'C:\Users\srira\OneDrive\Desktop\project\ratings_Electronics.csv', header=None)
df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']  # Adding column names
df = df.drop('timestamp', axis=1)  # Dropping timestamp

# Convert the 'rating' column to numeric, setting errors to NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')


# Drop rows with NaN values in the 'rating' column
df = df.dropna(subset=['rating'])

# Verify the changes
df.info()

# Filtering users who have rated 50 or more items
counts = df['user_id'].value_counts()
df_final = df[df['user_id'].isin(counts[counts >= 50].index)]

print('The number of observations in the final data =', len(df_final))
print('Number of unique USERS in the final data = ', df_final['user_id'].nunique())
print('Number of unique PRODUCTS in the final data = ', df_final['prod_id'].nunique())

# Creating the interaction matrix of products and users based on ratings and replacing NaN value with 0
final_ratings_matrix = df_final.pivot(index='user_id', columns='prod_id', values='rating').fillna(0)
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

# Finding the number of non-zero entries in the interaction matrix 
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)

# Finding the possible number of ratings as per the number of users and products
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)

# Density of ratings
density = (given_num_of_ratings / possible_num_of_ratings)
density *= 100
print('density: {:4.2f}%'.format(density))

final_ratings_matrix.head()

# Convert the matrix to a sparse matrix format
final_ratings_sparse = csr_matrix(final_ratings_matrix.values)

# Singular Value Decomposition
U, s, Vt = svds(final_ratings_sparse, k=50)  # here k is the number of latent features

# Construct diagonal array in SVD
sigma = np.diag(s)

# Predict the ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Create a DataFrame for the predicted ratings
preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=final_ratings_matrix.columns)
preds_df.head()

# Convert the predicted ratings DataFrame to a sparse matrix format
preds_matrix = csr_matrix(preds_df.values)

def recommend_items(user_index, interactions_matrix, preds_matrix, num_recommendations):
    # Get the user's ratings from the actual and predicted interaction matrices
    user_ratings = interactions_matrix[user_index,:].toarray().reshape(-1)
    user_predictions = preds_matrix[user_index,:].toarray().reshape(-1)

    # Creating a dataframe with actual and predicted ratings columns
    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Products'] = np.arange(len(user_ratings))
    temp = temp.set_index('Recommended Products')
    
    # Filtering the dataframe where actual ratings are 0 which implies that the user has not interacted with that product
    temp = temp.loc[temp.user_ratings == 0]   
    
    # Recommending products with top predicted ratings
    temp = temp.sort_values('user_predictions', ascending=False)  # Sort the dataframe by user_predictions in descending order
    print('\nBelow are the recommended products for user(user_id = {}):\n'.format(user_index))
    print(temp['user_predictions'].head(num_recommendations))

# Enter 'user index' and 'num_recommendations' for the user
recommend_items(121, final_ratings_sparse, preds_matrix, 5)
recommend_items(100, final_ratings_sparse, preds_matrix, 10)
