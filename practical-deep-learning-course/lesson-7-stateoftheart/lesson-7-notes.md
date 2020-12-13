# Lesson 7: Training a State-of-the-Art Model

## Normalization
- helps if input data is normalized, mean=0, sd=1 for training
- grabbing batch of data and averagin over all axes except the channel axis
  ```
  x, y = dls.one_batch()
  x.mean(dim=[0,2,3])  # TensorImage([0.482, 0.4711, 0.4511], device='cuda:5')
  x.std(dim=[0,2,3])  # TensorImage([0.2873, 0.2893, 0.3110], device='cuda:5')
  ```
- `Normalize` transform
  - normalizes data
  - acts on whole mini-batch at once, can add to `batch_tfms` in `DataBlock`
  - pass to transform mean and sd we want to use
    - by default fastai automatically calculates statistics from single batch of data
  ```
  def get_dls(bs, size):
      dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_image_files,
                    get_y=parent_label,
                    item_tfms=Resize(460),
                    batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                                Normalize.from_stats(*imagenet_stats)])
      return dblock.dataloaders(path, bs=bs)
  dls = get_dls(64, 224)
  x,y = dls.one_batch()
  x.mean(dim=[0,2,3])  # TensorImage([-0.0787, 0.0525, 0.2136], devicee'cuda:5')
  x.std(dim=[0,2,3])  # TensorImage([1.2330, 1.2112, 1.3031], device='cuda:5'))
  model = xresnet50()
  learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
  learn.fit_one_cycle(5, 3e-3)
  ```
    - here we use 'imagenet_stats' but if we don't pass any statistics to the 'Normalize' transform, fastai will automatically calculate them from a single batch of the data
- normalization is important when using pretrained models
  - pretrained model only knows how to work with type of data it has seen before
  - if average pixel value was 0 in data model was trained with but data has 0 as min possible value of pixel, the model will see something very different than what was intended
  - when a model is distributed, also need to distriubute statistics used for normalization since anyone using model for inference or transfer learning will need to use the same statistics
  - if using model from someone else, need to match normalization statistics that they used
    - when using pretrained model through `cnn_learner` fastai automatically adds the proper `Normalize` transform
    - only applies to pretrained models so we need to add normalization information manually when training from scratch
    
## Progressive Resizing
- we have been training with size 224, but we can begin training at a smaller size before that
- progressive resizing
  - gradually using larger and larger images as you train
  - start training using small images, end training using large images
  - spending most of epochs training with small images helps training complete faster
  - completing training with large images makes final accuracy higher
- even though the features of the images don't change when we increase the size of the images, there are still differences between the small and big images and so we would expect our model to perform slightly worse compared to if we didn't change anything at all
  - this is a lot like transfer learning, as our model is learning something slightly different compared to what it learned before
  - we should be able to use `fine_tune` method after resizing images
- progressive resizing is another form of data augmentation and so we would expect to see better generalization of our models that are trained with progressive resizing
- steps of progressive resizing:
  - first create `get_dls` function as above which takes image size and batch size and returns a `DataLoaders`
  - then create `DataLoaders` with a small size and use `fit_one_cycle` as usual, with less epochs than otherwise
  - then replace `DataLoaders` with modified image size and batch size and `fin_tune`
  ```
  dls = get_dls(128, 128)
  learn = Learner(dls, xresnet50(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
  learn.fit_one_cycle(4, 3e-3)
  learn.dls = get_dls(64, 224)
  learn.fine_tune(5, 1e-3)
  ```
- we can increase size of images as much as we'd like, but there will be no benefit of using image larger than the size of the images on disk
- progressive resizing may hurt performance for transer learning
  - happens if pretrained model is very similar to transfer learning task and dataset was trained on similar-size images so weights wouldn't need to be changed much in this case
  - training on smaller images may damage pretrained weights in this case
  - if transfer learning task uses images of different sizes, shapes, or styels than those in pretraining task, progressive resizing will probably help
 
## Test Time Augmentation
- we have only applied data augmentation on the training set so far, but we can also try making predictions for a few augmented versions of the validation set and averaging them
- fastai random cropping automatically uses center cropping for the validation set
  - it selects largest square area possible from the center of the image without going past the image's edges
  - for multi-label dataset this can be a problem when there are small objects towards the edges of an image that could be entirely cropped out by center cropping
  - we could avoid random cropping and resort to squishing and stretching the images to fit into a square space, but we miss out on a useful data augmentation and make the image recognition more difficult for the model since it has to learn incorrectly proportioned images
- test time augmentation (TTA)
  - better solution is to not just center crop for validation, but take a select number of areas to crop from the original rectangular image, pass each through the model, and take the maximum or average of the predictions
  - we can do this not just for different crops, but for different values across all of oru test time augmentation parameters
  - during inference or validation, we create multiple versions of each image using data augmentation, then take the average or maximum of the predictions for each augmented version of the image
- tta can improve accuracy a lot, but it will take longer for validation or inference (n times slower for n images of tta)
- by default, fastai uses unaugmented center crop image and four randomly augmented images for tta
- pass any `DataLoader` to fastai's `tta` method and it uses the validation set by defualt
  ```
  preds, targs = lern.tta()
  accuracy(preds, targs).item()  # 0.8737
  ```

## Mixup
- data augmentation technique that can provide dramatically higher accuracy especially when there isn't much data and there isn't a pretrained model trained on data similar to a particular dataset
- for each image:
  - select another image from your dataset at random
  - pick a weight at random
  - take a weighted average using the weight that was picked of the selected image with your image
    - this is the independent variable
  - take a weighted average using the same weight of this image's labels with your image's labels
    - this is the dependent variable
- psuedocode (`t` is weight for the weighted average)
  ```
  image2, target2 = dataset[randint(0, len(dataset)]
  t = random_float(0.5, 1.0)
  new_image = t * image1 + (1-t) * image2
  new_target = t * target1 + (1-t) * target2
  ```
- targets must be one-hot encoded for this to work
- ex:
  - say we have 10 classes and image i is at index 2 and image j is at index 7, the one hot encoded representations are then
    ```
    [0,0,1,0,0,0,0,0,0,0] and [0,0,0,0,0,0,0,1,0,0]
    ```
  - t = 0.3 then the target would be 
    ```
    [0,0,0.3,0,0,0,0,0.7,0,0]
    ```
- fastai does this by adding a *callback* to our `Learner`
  - `Callback`s inject custom behavior in the training loop
  - specified by the `cbs` parameter in `Learner`
- training a model with Mixup
  ```
  model = xresnet50()
  learn = Learner(dls, model, loss_func=CrossEntropyFlat(), metrics=accuracy, cbs=Mixup)
  learn.fit_one_cycle(5, 3e-3)
  ```
- with mixup, it's harder to train since it's harder to see what is in each image
  - model also has to predict two labels per image, rather than just one and figure out how much each one is weighted
- overfitting is less likely to be a problem in mixup since we're not showing the same image in each epoch, but instead we are showing a random combination of two images
- mixup requires many more epochs to get better accuracy compared to other augmentation approaches
  - > 80 epochs for Imagenette
- can be applied to data other than photos
  - mixup alos used for activations inside models rather than just inputs which allows it to be used for NLP and other data types as well
- with sigmoid and softmax, the labels are always between 0 and 1 exclusive, but our labels are 1s and 0s and so the loss can never be perfect
  - with mixup, the labels can be exactly 1 or 0 if the images that are 'mixed' happen to be of the same class
  - the rest of the time the labels are a linear combination such as 0.3 and 0.7
  - problem is mixup may 'accidentally' make labels bigger than 0 or smaller than 1 in this case

## Label Smoothing
- handles the problem of having to change the amount of mixup to get labels closer to or further from 0 and 1
- our model is often overconfident, gets gradients and learns to predict activations with higher and higher confidence which encourages overfitting
  - results in a model that always says 1 for a predicted category even if it isn't too sure because it was trained this way
- label smoothing
  - we encourage the model to be less confident which makes training more robust even if there is mislabeled data
  - results in a model that generalizes better
  - we replace all our 1s with a number a bit less than 1 and our 0s with a number a bit more than 0 and then train
- in practice
  - begin with one-hot-encoded labels
  - replace all 0s with epsilon/N
    - epsilon: a parameter (usually 0.1 meaning 10% unsure of labels)
    - N: number of classes
  - we want the labels to add up to 1 and so we replace the 1 by 1 - epsilon + epsilon/N
  - ex: for 10 classes where target corresponds to index 3 with epsilon=0.1
    ```
    [0.01, 0.01, 0.01, 0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    ```
  - implementing with fastai
    ```
    model = xresnet50()
    learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)
    learn.fit_one_cycle(5, 3e-3)
    ```
  - label smoothing also takes many epochs to see significant improvements
  
  
  
  
  
  
  
  
  
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
