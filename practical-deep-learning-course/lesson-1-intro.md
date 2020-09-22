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
