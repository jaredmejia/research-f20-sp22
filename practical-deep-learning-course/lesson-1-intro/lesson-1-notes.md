# Notes from Lesson 1: Introduction
* fastai cnn_learner function
  * used for image recognition
  * utilizes a convolutional neural network
  * parameter ```pretrained``` defaults to ```True```
    * makes use of a pretrained model that has been trained on recognized photos from the ImageNet dataset
    * pretrained models allow for more accurate models, more quickly, with less data
    * using pretrained models for a task different to what it was originally trained for = transfer learning
      * we may end up using transfer learning in two ways
        * to learn how to classify situations the robot encounters in the simulation
        * to learn how to transfer simulation learning to physical world computer vision (transfer learning of transfer learning?)
  * ex: ```learn = cnn_learner(dls, resnet34, metrics = error_rate)```
    * ```34``` in ```resnet34``` refers to the number of layers in variant of the resnet architecture
* ```fine_tune``` method
  * key to deep learning
    * determining how to fit the parameters of a model to get it to solve your problem
    * requires you to know the number of epochs
  * default ```fine_tune``` steps:
    * use one epoch to fit just the parts of the model needed to get the new random head to work correctly with the dataset
    * use the number of epochs rquested when calling the method to fit the entire model, updating the weights of the later layers faster than the earlier layers
* deep learning results aren't actually 'black box' models as most believe them to be
* image recognizers taking on non-image tasks
  * turns out a lot of things can be represented as images
    * ex: sound converted to spectrogram for sound detection
    * ex: malware to binary -> binary to 8 bit vector -> 8 bit vector to grayscale image
    * any classification task that can be converted to (distinct) images can be solved using image recognizers
* generalizing vs overfitting
  * we want a model that learns general lessons from our data which also apply to new items it will encounter, so taht it can make good predictions on thos items
  * we don't want to simply memorize what we have already seen and make poor predictions about new images (overfitting)
* segmentation
  * creating a model that can recognize the content of every individual pixel in an image
