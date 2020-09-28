# Notes from Lesson 2: From Model to Production
* the state of deep learning
  * computer vision
    * object recognition
      * computers can recognize what items are in an image at least as well as people can
    * object detection
      * deep learning allows for recognizing where objects in an image are
      * can highlight their locations and name each found object
      * variant of this is segmentation
        * when every pixel is categorized based on what kind of object is is
    * deep learning struggles
      * bad at recognizing images that are significantly different in structure or style to those trained in the model
        * ex: black and white images, hand-drawn images
        * must check for out-of-domain data
      * image labelling is slow/expensive
        * data augmentation sometimes helps with this
## Questionnare
* Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
* Where do text models currently have a major deficiency?
* What are possible negative societal implications of text generation models?
* In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
* What kind of tabular data is deep learning particularly good at?
* What's a key downside of directly using a deep learning model for recommendation systems?
* What are the steps of the Drivetrain Approach?
* How do the steps of the Drivetrain Approach map to a recommendation system?
* Create an image recognition model using data you curate, and deploy it on the web.
* What is DataLoaders?
* What four things do we need to tell fastai to create DataLoaders?
* What does the splitter parameter to DataBlock do?
* How do we ensure a random split always gives the same validation set?
* What letters are often used to signify the independent and dependent variables?
* What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
* What is data augmentation? Why is it needed?
* What is the difference between item_tfms and batch_tfms?
* What is a confusion matrix?
* What does export save?
* What is it called when we use a model for getting predictions, instead of training?
* What are IPython widgets?
* When might you want to use CPU for deployment? When might GPU be better?
* What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
* What are three examples of problems that could occur when rolling out a bear warning system in practice?
* What is "out-of-domain data"?
* What is "domain shift"?
* What are the three steps in the deployment process?0
