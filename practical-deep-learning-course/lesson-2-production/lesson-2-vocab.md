# Vocabulary from Lesson 2: From Model to Production
* data augmentation
  * synthetically generating variations of input in order to expand a dataset
* `DataLoaders`
  * fastai class that stores multiple `DataLoaders` objects you pass to it, normally a `train` and a `valid`
* drivetrain approach
  * method used as a tool to ensure that your modeling work is useful in practice
  * defined objective, levers, data, model
* loss
  * number that is higher if the model is incorrect (especially if it's also confident of its incorrect answer) or if it is correct, but not confident of its correct answer
* object detection
  * recognizing where objects in an image are
  * sometimes includes highlighting the locations of objects and labeling with their names
* object recognition
  * recognizing what items are in an image
* probability (loss context)
  * the confidence level, from zero to one, that the model assigned to its prediction
