# Lesson 5: Image Classification
### Data Labels with Regex
- `RegexLabeller` class for labeling with regular expressions
  ```
  pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    splitter=RandomSplitter(seed=42),
                    get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                    item_tfms=Resize(460),
                    batch_tfms=aug_transforms(size=224, min_scale=0.75))
  dls = pets.dataloaders(path/"images")
  ```
### Presizing
- want all images to have same dimensions so they can be collated into tensors being passed to the GPU
- want to minimize number of distinct augmentation computations performed
- presizing strategy:
  - resize images to relatively 'large' dimensions, much larger than target training dimensions
    - allows for further augmentation transforms on inner regions without creating empty zones
  - compose all common augmentation operations (including a resize to the final target size) into one, and perform the combined operation on the GPU only once at the end of processing, rather than performing the operations individually and interpolating multiple times
- with fastai
  - Crop full width or height
    - in `item_tfms`, applied to each individual image before copied to GPU
    - on training set crop area chosen randomly
    - on validation set, center square of image always chosen
    - use `Resize` as an item transform with a large size
  - Random crop and augment
    - in `batch_tfms`, applied to a batch all at once on the GPU
    - on training set, random crop and any other augmentations done first
    - on validation set, only resize to final size needed for model done
    - use `RandomResizedCrop` as a batch transform with a smaller size
      - added if `min_scale` parameter included in `aug_transforms` function
      - can also use `pad` or `squish` instead of `crop` for initial `Resize`

### Cross-Entropy Loss
- works even when dependent variable has more than two categories, results in faster and more reliable training
- `DataLoaders.one_batch()`
  ```
  x, y = dls.one_batch()
  ```
  - returns the dependent and independent variables as a mini-batch
  - independent variable tensor has as many rows as the batch size
- `Learner.get_preds` to view predictions (activations of the final layer of our network)
  ```
  preds, _ = learn.get_preds(dl=[(x,y)])
  ```
  - either takes a dataset index (0 for train, 1 for valid) or an iterator of batches
  - returns predictions and targets by default
- Softmax
  - `def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)`
  - returns values that are all between 0 and 1 and sum to 1 (probability distribution)
- log likelihood
  - `nll_loss` (negative log likelihood loss)
    - the mean of the positive or negative log of our probabilities (depending on whether it's the correct or incorrect class)
    - assumes we already took the log of the softmax
    - PyTorch `log_softmax` combines `log` and `softmax`
    - `nll_loss` is to be used after `log_softmax`
- `nn.CrossEntropyLoss()` does `log_softmax` and then `nll_loss`

### Unfreezing and Transfer Learning
- fine tuning
  - we want to replace the random weights in our added linear layers which in transfer learning that correctly achieve the desired task without breaking the carefully pretrained weights and other layers
  - we tell the optimizer to only update the weights in the randomly added final layers and we don't change the rest of the neural network at all 
    - this is freezing the pretrained layers
  - when creating a model from a pretrained network fastai automatically freezes all of the pretrained layers for us
  - the `fine_tune` method
    - trains the randomly added layers for one epoch, with all other layers frozen
    - unfreezes all of the layers, and trains them all for the number of epochs requested
    - contains various parameters to let you control this functionality
  - to unfreeze model `Learn.unfreeze()`
  - should find the best learning rate again after unfreezing since having more layers to train and weights that have already been trained for three epochs means the previous learning rate isn't appropriate anymore `Learn.lr_find()`
    - note in a pretrained model, we don't look for the point with the maximum gradient

### Discriminative Learning Rates
- the deepest layers of our pretrained model might not need as a high a learning rate as the last ones, so we shoudld use a different lr for those
- we use a lower learning rate for early layers of the neural network and a higher learning rate for the layers (especially the randomly added layers)
- `slice` object passed to learning rate
  - first value passed will be the lr in the earliest layer of the network
  - second value passed will be the lr in the final layer
  - layers in between will have lrs multiplicatively equidistant throughout the range
    ```
    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.fit_one_cycle(3, 3e-3)
    learn.unfreeze()
    learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
    ```

### Selecting the Number of Epochs
- first approach to training: pick number of epochs that will train in amount of time you have
- often metrics get worse at the end of training
  - validation loss gets worse first during training due to overconfidence
  - validation loss gets even worse once it begins incorrectly memorizing data
- early stopping is when a model is saved at the end of each epoch and then the model with the best accuracy out of all the models is selected
  - unlikely to give best model since epochs in middle occur before lr has had chance to reach small values which is when the best result comes
  - if overfitting occurs, start over from scratch and select total number of epochs based on where previous best results were found

### Deeper Architectures
- a model with more parameters models the data more accurately (generally)
- bigger models able to better capture underlying relationships in data and also to capute and memorize specific details of individual imges
- deeper models require more GPU RAM
  - may need to lower batches
  - `bs=` param in `DataLoaders` allows you to specify batch size
- deeper models take longer
  - mixed precision training can help speed training up
    - uses less precise numbers (half-precision floating point, fp16) where possible in training
    - NVIDIA GPUs have *tensor cores* which speed up training by 2-3x
      - use much less GPU memory as well
      - enable in fastai with `to_fp16()` after `Learner` creation
        ```
        from fastai.callback.fp16 import *
        learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
        learn.fine_tune(6, freeze_epochs=3)  # training 3 epochs while frozen
        ```
        
    
    
    
    
    
    
    
    
    
    
