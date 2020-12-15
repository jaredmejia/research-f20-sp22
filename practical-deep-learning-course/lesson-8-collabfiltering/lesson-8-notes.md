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
  - ex: if the first latent user factor represents how much the user likes action moves and the first latent movie factor represents if the movie has a lot of action or not, the product of those will be particularly high if either the user likes action movies and the movie has a lot of action in it or if the user doesn't like action movies and the movie doesn't have any action in it
  - if there is a mismatch (user loves action movies but the movie isn't an action film, or the user doesn't like action movies and it is one) the product will be very low
- we finally calculate the loss
  - can use any loss function, i.e. mean squared error
  - with this we can optimize the parameters (latent factors) using stochastic gradient descent to minimize the loss
  - the optimizer calculates the match between each movie and each user using the dot product and compares to the actual rating the user gave
  - the derivative of this value is then used to step the weights by multiplying the derivativ eby the learning rate
- repeat

### Creating DataLoaders from taular data
- file `u.item` contains correspondence of IDs to titles
  ```
  movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',
                        uscols=(0,1), names=('movie','title'), header=None)
  ```
- we merge this with our `ratings` table to get the user ratings by title
  ```
  ratings = ratings.merge(movies)
  ```
- `DataLoaders` from table
  - by default takes first column for the user, second column for the item (movie title), third column for rating
  - need to change value of `item_name` parameter to use the movie titles for the second column
    ```
    dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
    ```
- need to represent movie and user latent factor tables as matrices in order to use PyTorch with them
  ```
  n_users = len(dls.classes['user'])
  n_movies = len(dls.classes['title'])
  n_factors = 5
  user_factors = torch.randn(n_users, n_factors)
  movie_factors = torch.randn(n_movies, n_factors)
  ```
- to calculate the result of a particular movie and user combination, we need to look up the indices of both the movie and the user in their respective latent factor matrices
  - we need to represent looking up an index as a matrix product
  - we replace our indices with one-hot-encoded vectors
  - when we multiply the matrix of the users by the one hot encoded vector of index i we get the same thing as the vector at index i in the matrix of users
    ```
    one_hot_3 = one_hot(3, n_users).float()
    user_factors.t() @ one_hot_3  # tensor([-0.4586, -0.9915, -0.4052, -0.3621, -0.5908])
    user_factors[3]  # tensor([-0.4586, -0.9915, -0.4052, -0.3621, -0.5908])
    ```
  - if we did this for a few indices at once, we could get a matrix of one-hot-encoded vectors, which would allow us to use matrix multiplication
  - this would work, but would take a lot of memory and time
  - there is no reason to store the one-hot-encoded vector or to search through it to find the occurrence of the number one when we could just index into an array directly with an integer
- embedding
  - deep learning libraries such as PyTorch include a special layer that indexes into a vector using an integer, but has its derivative calculated in such a way that it is identical to what it would have been if it had done a matrix multiplication with a one-hot-encoded vector
  - embedding is multiplying by a one-hot encoded matrix, using a computational shortcut so that it can be implemented by simply indexing directly
  - the embedding matrix is what we multiply the one-hot-encoded matrix by (or index into directly using the computational shortcut)
  - the embeddings represent the features that seem important to the relations between users and movies

### Collaborative Filtering from Scratch
- if we create a PyTorch module any time that module is called, PyTorch will call a method in the class called `forward` and will pass along to it any parameters included in the call
  ```
  class DotProduct(Module):
      def __init__(self, n_users, n_movies, n_factors):
          self.user_factors = Embedding(n_users, n_factors)
          self.movie_factors = Embedding(n_movies, n_factors)
          
      def forward(self, x):
          users = self.user_factors(x[:,0])
          movies = self.movie_factors(x[:,1])
          return (users * movies).sum(dim=1)
  ```
- note: input of model is a tensor of shape `batch_size x 2` where the first column `x[:, 0]` contains user IDs and the second column `x[:, 1]` contains the movie IDs
- training the model
  ```
  model = DotProduct(n_users, n_movies, 50)
  learn = Learner(dls, model, loss_func=MSELossFlat())
  learn.fit_one_cycle(5, 5e-3)
  ```
- to improve the model, we need to force the predictions to be between 0 and 5
  - we use `sigmoid_range` for this
  - empirically, it is found to be better to have the range go a little over 5, so we use `(0, 5.5)`
  ```
  class DotProduct(Module):
      def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
          self.user_factors = Embedding(n_users, n_factors)
          self.movie_factors = Embedding(n_movies, n_factors)
          self.y_range = y_range
      
      def forward(self, x):
          users = self.user_factors(x[:,0])
          movies = self.movie_factors(x[:,1])
          return sigmoid_range((users * movies).sum(dim=1), *self.y_range)
  ```
- training again
  ```
  model = DotProduct(n_users, n_movies, 50)
  learn = Learner(dls, model, loss_func=MSELossFlat())
  learn.fit_one_cycle(5, 5e-3)
  ```
- at this point we only model the types of movies and don't take into account the fact that some users are more positive than others and some movies are better than others
  - we need to introduce biases in addition to our weights in order to model this
    ```
    class DotProductBias(Module):
        def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
            self.user_factors = Embedding(n_users, n_factors)
            self.user_bias = Embedding(n_users, 1)
            self.movie_factors = Embedding(n_movies, n_factors)
            self.movie_bias = Embedding(n_movies, 1)
            self.y_range = y_range
            
        def forward(self, x):
            users = self.user_factors(x[:,0])
            movies = self.movie_factors(x[:,1])
            res = (users * movies).sum(dim=1, keepdim=True)
            res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
            return sigmoid_range(res, *self.y_range)
    ```
  - training once more
    ```
    model = DotProductBias(n_users, n_movies, 50)
    learn = Learner(dls, model, loss_func=MSELossFlat())
    learn.fit_one_cycle(5, 5e-3)
    ```

### Weight Decay
- to handle overfitting with collaboration filtering, we can't use data augmentation
- weight decay, or L2 regularization consists of adding to the loss ufnction of the sum of all weights squared
  - then, when we compute the gradients, this results in an additional contriubtion to them that encourages the weights to be as small as possible
  - it tends to be the case that the larger the coefficients, the steeper the 'canyons' in our loss function
  - letting the model learn high parameters may cause the model to fit all data points in the training set with an overcomplex function with sharp changes which will lead to overfitting
  - limiting weights from growing too much hinders the training of the model but yields a state where it generalizes better
  - mathematically:
    ```
    loss_with_wd = loss + wd * (parameters**2).sum()
    parameters.grad += wd * 2 * parameters
    ```
  - we can skip computing the sum and just take into account the gradient `wd * 2 * parameters`, but we just make `wd` twice as big in practice since it is a parameter we choose
- training with weight decay yields less overfitting
  ```
  model = DotProductBias(n_users, n_movies, 50)
  learn = Learner(dls, model, loss_func=MSELossFlat())
  learn.fit_one_cycle(5, 5e-3, wd=0.1)
  ```

### Creating Embedding Module
- we need a randomly initialized weight matrix for each of the embeddings
- to tell `Module` we want to treat a tensor as a parameter, we need to wrap it in the `nn.Parameter` class
  - this class automatically calls `requires_grad_` for us and is used as a marker to show what to include in `parameters`
- function to create a tensor as a parameter, with random initialization
  ```
  def create_params(size):
      return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))
  ```
- creating `DotProductBias`, but without `Embedding`
  ```
  class DotProductBias(Module):
      def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
          self.user_factors = create_params([n_users, n_factors])
          self.user_bias = create_params([n_users])
          self.movie_factors = create_params([n_movies, n_factors])
          self.movie_bias = create_params([n_movies])
          self.y_range = y_range
          
      def forward(self, x):
          users = self.user_factors[x[:,0]]
          movies = self.movie_factors[x[:,1]]
          res = (users*movies).sum(dim=1)
          res += self.user_bias[x[:,0]] + self.movie_bias[x[:,1]]
          return sigmoid_range(res, *self.y_range)
  ```
- training again
  ```
  model = DotProductBias(n_users, n_movies, 50)
  learn = Learner(dls, model, loss_func=MSELossFlat())
  learn.fit_one_cycle(5, 5e-3, wd=0.1)
  ```

### Interpreting Embeddings and Biases
- getting movies with lowest values in bias vector
  ```
  movie_bias = learn.model.movie_bias.squeeze()
  idxs = movie_bias.argsort()[:5]
  [dls.classes['title'][i] for i in idxs]  # list of 5 movie titles
  ```
  - for these movies, even when a user is very well matched to its latent factors, they still generally don't like it
- getting movies with highest bias
  ```
  idxs = movie_bias.argsort(descending=True)[:5]
  [dls.classes['title'][i] for i in idxs]
  ```

### Using fastai.collab
- fastai `collab_learner`
  ```
  learn = collab_learner(dls, n_factors=50, y_range(0, 5.5))
  learn.fit_one_cycle(5, 5e-3, wd=0.1)
  ```
- can take a look at the layers by printing the model
  ```
  learn.model
  ```
- can again analyze the results of the learned biases in our model
  ```
  movie_bias = learn.model.i_bias.weight.squeeze()
  idxs = movie_ibas.argsort(descending=True)[:5]
  [dls.classes['title'][i] for i in idxs]  # list of movie titles
  ```
 
### Embedding Distance
- we can use L2 distance to calculate the distance between two embeddings
- if two movies are very similar, their embedding vectors would be very similar and so the users that like them would be nearly the same
- movie similarity can be defined by the similarity of users that like the movies
- therefore, the distance between two movies' embedding vectors define the similarity
- finding most similar movie to Star Wars
  ```
  movie_factors = learn.model.i_weight.weight
  idx = dls.classes['title'].o2i['Star Wars (1977)']
  distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
  idx = distances.argsort(descending=True)[1]
  dls.classes['title'][idx]  # 'Empire Strikes Back, The (1980)`
  ```
  
### Deep Learning for Collaborative Filtering
- our model using dot products to approach collaborative filtering is *probabilistic matrix factorization* (PMF)
- Deep Learning is an alternative that works similarly given the same data 
- we first need to take the results of the embedding lookup and concatenate the activations together
- fastai `get_emb_sz` function returns recommended sizes for embedding matrices for data
  ```
  embs = get_emb_sz(dls)
  embs  # [(944, 74), (1635, 101)]
  ```
- implementing the class for deep learning collaborative filtering
  ```
  class CollabNN(Module):
      def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
          self.user_factors = Embedding(*user_sz)
          self.item_factors = Embedding(*item_sz)
          self.layers = nn.Sequential(
              nn.Linear(user_sz[1]+item_sz[1], n_act),
              nn.ReLU(),
              nn.Linear(n_act, 1))
          self.y_range = y_range
          
      def forward(self, x):
          embs = self.user_factors(x[:,0]), self.item_factors(x[:,1])
          x = self.layers(torch.cat(embs, dim=1))
          return sigmoid_range(x, *self.y_range)
  ```
- training
  ```
  model = CollabNN(*embs)
  learn = Learner(dls, model, loss_func=MSELossFlat())
  learn.fit_one_cycle(5, 5e-3, wd=0.01)
  ```
- fastai provides this model in `fastai.collab` if we pass `use_nn=True` to `collab_learner`
  - also automatically calls `get_emb_sz`
  ```
  learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
  learn.fit_one_cycle(5, 5e-3, wd=0.1)
  ```
- `EmbeddingNN` class
  ```
  @delegates(TabularModel)
  class EmbeddingNN(TabularModel):
      def __init__(self, emb_szs, layers, **kwargs):
          super().__init__(emb_szs, layers=layers, n_cont=0, out_sz=1, **kwargs)
  ```
  - inherits from the `TabularModel` class which gives it all its functionality
