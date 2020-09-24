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
* tabular models
  * predicting one column of a table based on information in other columns of the table
  * need to tell fastai which columns are categorical
    * which columns contain values that are of a discrete set of choices
    * ex: ```cat_names = ['workclass', 'education', 'marital-status', occupation']```
  * need to specify which columns are continuous
    * which columns contain a number that represents a quantity
    * ex: ```cont_names = ['age', 'years-of-education']```
  * usually aren't any pretrained models for tabular modeling
    * can't use ```fine_tune```
      * though fastai does let you use ```fine_tune``` even without pretraining in some cases
    * use ```fit_one_cycle```
      * method used for training fastai models from scratch—without transfer learning
  * ```show_results``` method gives examples of predictions
* using datasets
  * in practice, best to do most experimentation and prototyping with subsets of data
  * only use full dataset once you have a good understanding of what needs to be done
  * hyperparameters
    * choices made on parameters about parameters
    * choices on network architecture, learning rates, data augmentation, etc.
  * training, validation, and test sets
    * test and validation sets need to have enough data to be good estimates of accuracy
    * validation set is for tweaking hyperparameters
    * test set is for evaluating model after all changes to the model, and is not to be used to improve the model
    * can do better than randomly grabbing a fraction of the original dataset for a test set or validation set
      * validation and test sets must be representative of new data we will see in the future
      * time series data
        * randomly choosing a subset of the data will not be representative of what you will see in the future
        * better to choose a continuous section with the latest dates as the validation and test sets
          * this allows you to predict going forward, rather than getting a sense of the data without any relation to time
          * can use back-testing to check whether models are predictive of futre periods
      * case when data for predictions will be qualitatively different from the data you are training with
        * especially important when datasets are not very diverse
        * if person/object is in both training and validation sets, it will be easy for model to predict
        * much better to have people/objects not in training sets in validation and training sets
* GPU runs faster than CPU for deep learning
