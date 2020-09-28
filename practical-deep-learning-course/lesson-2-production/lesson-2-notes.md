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
    * many problems can be turned into a computer vision one
  * text (natural language processing)
    * computers are good at classifying documents based on categories
      * categories may include spam, sentiment, author, source website, etc.
    * text generation
      * very good at generating context-appropriate text
        * replies to social media posts, imitating an author's style, etc.
      * not very good at generating correct responses
        * dangerous as computer generating content may appear compelling, yet not actually be correct
    * text generation models will always be technologically ahead of models recognizing automatically generated text
      * a model recognizing a artificially generated content could be used to improve the generator creating the content
    * translation
    * summarization
    * counting
    * protein analysis
  * combining text and images
    * caption generating for images
      * no guarantee captions will be correct
  * tabular data
    * deep learning may not perform much better than random forests and gradient boosting machines at this point
      * deep learning takes longer to train than random forests and gradient boosting machines at this point
    * deep learning greatly increases the variety of columns that can be included in tabular data
      * ex: natural language and high-cardinality categorical columns
  * recommendation systems
    * a special type of tabular data
      * high-cardinality categorical variable representing users, and another representing products (or something similar)
* the drivetrain approach
  * using data to produce actionable outcomes
  * basic idea
    * define clear objective
      * what outcome you are trying to achieve
    * levers
      * what inputs you can control
      * what actions you can take to better achieve the objective
    * data
      * what data you can collect
    * build a model you can use to determine the best actions to get the best results in terms of your objective
* gathering data
  * 
## Questionnare
* Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
* Where do text models currently have a major deficiency?
  * correctness
* What are possible negative societal implications of text generation models?
  * text generation could be used on. massive scale to spread disinformation, create unrest, and encourage conflict
* In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
  * processes in which the model and a human user interact closer are much safer than entirely automated processes yet still can be much more productive than human input alone
* What kind of tabular data is deep learning particularly good at?
  * recommendatin systems since deep learning is good at handling high-cardinality categorical variables
* What's a key downside of directly using a deep learning model for recommendation systems?
  * they only tell you what kinds of products a user might like
  * don't tell you what recommendations would be helpful for a user
* What are the steps of the Drivetrain Approach?
  * defined objective, levers, data, models
* How do the steps of the Drivetrain Approach map to a recommendation system?
  * objective: additional sales through relevant recommendations
  * lever: ranking of recommendations
  * data: data to generate recommendations from a wide range of customers
  * models: models for purchase probabilities, one conditional on seeing a recommendation and one conditional on not seeing a recommendation
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
