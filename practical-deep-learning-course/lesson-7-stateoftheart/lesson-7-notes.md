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
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
