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
  - `splitter` function
    - rather than randomly splitting into test and validation, explicitly chooses elements of validation set
    ```
    def splitter(df):
        train = df.index[~df['is_valid']].tolist()
        valid = df.index[df['is_valid']].tolist()
        return train, valid
    ```
- `DataLoader` collates items from a `Dataset` into a mini-batch
  - tuple of tensors where each tensor stacks the items from the location in the `Dataset` item
  ```
  dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                    splitter=splitter,
                    get_x=get_x,
                    get_y=get_y,
                    item_tfms = RandomResizedCrop(128, min_scale=0.35))
  dls = dblock.dataloaders(df)
  ```

### Binary Cross-Entropy
- `Learner` object contains 4 main things
  - the model
  - a `DataLoaders` object,
  - an `Optimizer`,
  - the loss function to use
- getting model activations
  ```
  learn = cnn_learner(dls, resent18)
  x, y = to_cpu(dls.train.one_batch())  # getting inputs and outputs
  activs = learn.model(x)  # getting activations for input
  activs.shape  # torch.Size([64, 20]) -> batch size = 64, num categories = 20
  activs[0]  # the activations of the 20 categories for the first element of the batch
  ```
- binary cross-entropy function
  - with one-hot-encoded dependent vairable can't use `nll_loss` or `softmax` and hence can't use `cross_entropy`
    - `softmax`
      - requires all predictions to sum to 1
      - pushes one activation much larger than others
      - we may have multiple objects that appear in an image, so restricting max sum of activations to 1 doesn't work for multi-label classify
      - we also may want sum to be less than 1 if we don't think any categories appear in the image
    - `nll_loss`
      - returns value of a single activation corresponding with the single label for an item
      - woudln't work for multiple labels
  - `binary_cross_entropy` is `mnist_loss` with `log`
    ```
    def binary_cross_entropy(inputs, targets):
        inputs = inputs.sigmoid()
        return -torch.where(targets==1, inputs, 1-inputs).log().mean()
    ```
  - each activation is compared to each target for each column, so no need to make the function work for multiple columns
  - different versions for multi-label datasets
    - `F.binary_cross_entropy` and `nn.BCELoss`
      - calculate cross-entropy on a one-hot-encoded target
      - don't include initial `sigmoid`
    - `F.binary_cross_entropy_with_logits` and `nn.BCEWithLogitsLoss`
      - include iniital `sigmoid` and do binary cross-entropy in a single function
  - for single-label datasets
    - `F.nll_loss` or `nn.NLLLoss`
      - versions without initial softmax
    - `F.cross_entropy` or `nn.CrossEntropyLoss`
      - versions with initial softmax
  - fastai will use `nn.BCEWithLogitsLoss` by default if `DataLoaders` has multiple category labels
    - don't need to explicitly tell fastai to use this loss function
- accuracy function for multiple labels
  - can't use previous accuracy function since it simply chose the category with the highest activation
  - after applying sigmoid to activations, we need to determine which ones are 0s and which are 1s by picking a threshold
  - values above the threshold are considered a 1, values below are considered a 0
    ```
    def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
        if sigmoid: inp = inp.sigmoid()
        return ((inp>thresh)==targ.bool()).float().mean()
    ```
  - `accuracy_multi` metric uses 0.5 as default value for `threshold`
    - Python `partial` function
      - allows us to bind a function with some arguments or keyword function
      - creates a new version of a function that whenever called, always includes the specified arguments
  - training model using `partial` function to set default of threshold to 0.2 for `accuracy_multi` metric
    ```
    learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
    learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
    ```
  - `Learner.validate()` returns the validation loss and metrics so we can determine how well we have done choosing our threshold
    ```
    learn.metrics = partial(accuracy_multi, thresh=0.1)
    learn.validate()
    ```
  - finding the best threshold level
    ```
    preds, targs = learn.get_preds()
    accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)
    xs = torch.linspace(0.05, 0.95, 29)
    accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
    plt.plot(xs, accs)
    ```
    - we can try multiple values of a threshold without overfitting since the relationship when changing the threshold results in a smooth curve
      


### Regression Problem
- sometimes our labels are one or several numbers, a quantity rather than a category










