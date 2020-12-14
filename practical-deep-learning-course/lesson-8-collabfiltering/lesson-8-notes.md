# Lesson 8: Collaborative Filtering Deep Dive
### Overview
- collaborative filtering: looking at products that current user has used or liked, find other users that have used or liked similar products, recommend other products that those users have used or liked
- more generally we have items rather than products
- latent factors: variables that are not directly observed but are inferred from other variables that are observed

### Getting data
- we have file *u.data* that is tab-separated and whose columns are user, movie, rating, timestamp, though the the names aren't encoded so we need to indicate them when reading the file with Pandas
  ```
  ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                        names=['user','movie', 'rating', 'timestamp'])
  ```
- 

### Learning the Latent Factors
- we first randomly initialize parameters that are the set of latent factors for each user and movie
  - need to decide how many to use
  - each user will have a set of the factors and each movie will have a set of the factors
- we then calculate predictions
  - simply take the dot product of the vector for each movie with the vector for each user
