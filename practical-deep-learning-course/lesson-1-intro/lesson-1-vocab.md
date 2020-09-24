# fast.ai course Vocabulary
* architecture
  * the template of the model we're trying to fit
  * the actual mathematical function we're passing the input data and parameters to
* Convolutional Neural Network (CNN)
  * a type of neural network that works particularly well for computer vision tasks
* epoch
  * one complete pass through the dataset
* fine-tuning
  * a transfer learning technique where the parameters of a pretrained model are updated by training for additional epochs using a different task to that used for 
* fit (or train)
  * update the parameters of the model such that the predictions of the model using the input data match the target labels
* head (of a model)
  * the part of the model that is newly added to be specific to the new dataset
pretraining
* hyperparameters
  * parameters about parameters
  * the higher-level choices that govern the meaning of the weight parameters
  * include choices on network architecture, learning rates, data augmentation strategies, etc.
* loss
  * a measure of how good a model is, chosen to drive training via stochastic gradient descent (SGD)
* model
  * the combination of the architecture with a particular set of parameters
* overfitting
  * training a model in such a way that it remembers specific features of the input data, rather than generalizing well to data not seen during training
* parameters
  * the values in the model that change what task it can do, and are updated through model training
* pretrained model
  * a model that has already been trained, generally using a large dataset, and will be fine-tuned
* segmentation
  * creating a model that can recognize the content of every individual pixel in an image
* tabular
  * data that is in the form of a table, such as from a spreadsheet, database, or CSV file
  * a tabular model is a model that tries to predict one column of a table based on information in other columns of the table
* test set
  * portion of data reserved only for the evaluation of the model after all modifications of the model
  * cannot be used to improve the model (overfitting)
* training set
  * the data used for fitting the model
  * does not include any data from the validation set
* transfer learning
  * using a pretrained model for a task different to what it was originally trained for
* validation set
  * a set of data held out form training, used only for measuring how good the model is
