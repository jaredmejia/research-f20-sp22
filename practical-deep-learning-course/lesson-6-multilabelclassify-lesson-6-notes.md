# Lesson 6: Multi-Label Classification

### Multi-Label Classification Problem
- sometimes we want to predict more than one label per image
- may be one or more kinds of object, or there may be no objects at all in the classes we're searching for

### Pandas and DataFrames
- Pandas is a Python library used to manipulate and analyze tabular and time series data
- Pandas `DataFrame` class
  - represents a table of rows and columns
  - can get a `DataFrame` from a CSV file, a database table, Python dictionaries, and other sources
  - rows and cols of `DataFrame` accessed with `iloc` property similar to matrices
    ```
    df.iloc[:,0]  # the first column of the DataFrame
    df.iloc[0,:]  # the first row of the DataFrame
    df.iloc[0]  # same as previous
    df['fname']  # grabbing a column by name
    ```
  - creating new columns and doing calculations on columns
    ```
    tmp_df = pd.DataFrame({'a':[1,2], 'b':[3,4]})
    tmp_df['c'] = tmp_df['a']+tmp_df['b']  # creating a column c, sum of cols a and b
    ```
    
### Constructing DataBlock from DataFrames
- fastai classes for representing and accessing a training set or validation set
  - `Dataset`: a collection that returns a tuple of the indpendent and dependent variable for a single item
  - `DataLoader`: an iterator that provides a stream of mini-batches, where each mini-batch is a tuple of a batch of independent variables and a batch of dependent variables
- fastai classes for brining training and validation sets together
  - `Datasets`: an object that contains a training `Dataset` and a validation `Dataset`
  - `DataLoaders`: an object that contains a training `DataLoader` and a validation `DataLoader`
- `DataBlock`s are built gradually step by step
- `DataBlock` with multiple labels for each item
  ```
  dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                    get_x = get_x, get_y = get_y)
  dsets = dblock.datasets(df)
  dsets.train[0]  # (PILImage mode=RGB size=500x375,
 TensorMultiCategory([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]))
  ```
  - rather than a a single binary integer, we now have a a list of zeros, and a one where the category is present
  - *one-hot encoding*
    - all 0s, except a 1 in the position for which each label occurs in the image
  - `torch.where(bool)` returns all indices where condition is met
  


### Regression Problem
- sometimes our labels are one or several numbers, a quantity rather than a category
