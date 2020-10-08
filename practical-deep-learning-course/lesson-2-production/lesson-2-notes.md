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
* jupyter features
  * press tab for autocomplete suggestions of a function or argument name
  * press shift and tab simulataneously inside the parentheses of a function to display a window with the signature of the function and a short description
    * pressing twice with expand documentation
    * pressing three times will open a full window with the same information at the bottom of your screen
  * typing ?func_name and executing will open a window with the signature of the function and a short description
  * typing ??func_name and executing will open a window with the signature of a function, a short description, and the source code
  * executing doc(func_name) will open a window with the signature of the function, a short description, and links to the source code on GitHub and the full documentation of the function in the library docs
    * only works with fastai library
  * executing %debug in the next cell will open the Python debugger
* working with data
 * `DataLoaders`
   * class that stores whatever `DataLoader` objects you pass to it and makes them available as `train` and `valid`
   * provides the data for your model
   * ```
     class DataLoaders(GetAttr):
         def __init__(self, *loaders): self.loaders = loaders
         def __getitem___(self, i): return self.loaders[i]
         train, valid = add_props(lambda i, self: self[i])
     ```
   * turning data into `DataLoaders` object:
     * what kinds of data we are working with
     * how to get the list of items
     * how to label these items
     * how to create the validation set
 * data block API
   * allows you to customize every stage of the creation of your `DataLoaders`
   * ex:
   * ```
     bears = DataBlock(
         blocks=(ImageBlock, CategoryBlock),
         get_items=get_image_files,
         splitter=RandomSplitter(valid_pct=0.2, seed=47),
         get_y=parent_label,
         item_tfms=Resize(128))
     ```
     * `blocks=(ImageBlock, CategoryBlock)`
       * specify what we want for the independent and dependent variables
         * indpendent variable (x): what we are using to make predictions from
         * dependent variabl (y)e: our target
     * `get_items=get_image_files`
       * underlying items of `DataLoaders` is a path
       * `get_image_files` takes a path, returns a list of al the images in that path
     * `splitter=RandomSplitter(valid_pct=0.2, seed=42)`
       * randomly splitting our training and validation sets
       * setting the seed means we will have the same training/validation split each time we run
     * `get_y=parent_label`
       * telling fastai which function to call to create the labels in the dataset
       * `parent_label` gets the name of the folder a file is in
     * `item_tfms=Resize(128)`
       * mini-batch: several images that we feed our model at a time
       * tensor: a group of mini-batches in a large array
       * all images need to be of the same size to be in a tensor
       * item transforms: pieces of code that run on each individual item
       * `Resize`: fastai predefined transform
     * `dls = bears.dataloaders(path)`
       * `DataLoaders` includes validation and training `DataLoader`s
       * `DataLoader`: class that provides batches of a few items at a time to the GPU
         * looping through a `DataLoader` will give you 64 (by defuault) items at a time stacked into a single tensor
 * `show_batch` method
   * allows us to take a look at a few items of a `DataLoader`
   * ex: `dls.valid.show_batch(max_n=4, nrows=1)`
 * `Resize`
   * by default crops the images to fit a square shape of the size requested using the full width or height
   * `ResizeMethod.Squish`: squishes or stretches the image to fit the size requested
   * `ResizeMethod.Pad`: pads the images with zeros (black) to fit the size requested
   * `RandomResizedCrop`
     * randomly selects part of an image and crops to just that part
     * on each epoch we randomly select a different part of each image
     * the model learns to focus on/recognize different features in the images
     * model begins to understand the basic concept of what an object is and how it can be represented in an image
     * ex: `bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))`
 * Data Augmentation
   * def: creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data
   * examples for images: rotation, flipping, perspective warping, brightness changes, and contrast changes
   * `aug_transforms` function
     * fastai function that works well for augmenting images
     * once images are all the same size, this function can be applied to entire batch of them using the GPU
       * saves a lot of time
     * `bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))`
       * `batch_tfms` parameter: tells fastai we want to use these transforms on a batch
       * `mult=2` doubles the amount of augmentation compared to the default (to see the difference)
* Training Model -> Cleaning Data
  ```
  bears = bears.new(item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())
  dls = bears.dataloaders(path)
  learn = cnn_learner(dls, resnet18, metrics=error_rate)
  learn.fine_tune(4)
  interp = ClassificationInterpretation.from_learner(learn)
  interp.plot_confusion_matrix()
  interp.plot_top_losses(5, nrows=1)
  ```
  * confusion matrix
    * rows represent the categories in the dataset
    * columns represent the images the model predicted within the categories
    * diagonal shows the images that were classified correctly
    * off-diagonal cells represent those classified incorrectly
  * `plot_top_losses`
    * shows images with the highest losses in the dataset
    * each image labeled with: prediction, actual (target label), loss, and probability
  * it is good to train a quick and simple model first
    * the model can help you find data issues more quickly and easily
    * data cleaning to remove data that shouldn't be included in training or to update bad labels
  * `ImageClassifierCleaner`
    * allows you to choose a category and the training or validation set and view the highest-loss images in order of highest to lowest loss
    * provides menus to allow images to be selected for removal or relabeling
    * to delete all images selected for deletion `for idx in cleaner.delete(): cleaner.fns[idx].unlink()`
    * to move images which have been selected for a different category: `for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
* Using the Model for Inference
  * `export` method
    * saves the architecture and trained parameters of a model
    * also saves the definition for how to create your `DataLoaders`
      * otherwise would have to redefine how to transform the data in order to use the model in production
      * fastai automatically uses validation set `DataLoader` for inference by default
        * data augmentation will not be applied
    * `learn.export()`
      * saves a 'pkl' file
    ```
    path = Path()
    path.ls(file_exts='.pkl')
    ```
      * checking file exists
  * inference: using a model for getting predictions rather than training
    * `learn_inf = load_learner(path/'export.pkl')`
      * creating an inference learner from the exported file
    * `learn_inf.predict('images/grizzly.jpg')`
      * getting a prediction for a single image (usually how it is done in inference)
      * returns three things
        * the predicted category in the same format originally provided
        * the index of the predicted category
          * based on the order of the categories in the stored list of all possible categories of the `DataLoaders`
        * the probabilities of each category
    * `learn_inf.dls.vocab`
      * accessing `DataLoaders` as an attribute of the `Learner`
  * IPython widgets (ipywidgets)
    * GUI components with JavaScript and Python functionality in a web browser
    * can be created and used within a Jupyter notebook
  * Voilà
    * system for making applications consisting of IPython widgets available to end users
    * users don't have to use Jupyter
    * helps automatically convert the notebook into a simpler, easier to deploy web app which functions like a normal web application rather than a notebook
  * building a GUI
    * `btn_upload = widgets.FileUpload()`
    * `btn_upload`
      * file upload widget
    ```
    img = PILImage.create(btn_upload.data[-1])
    out_pl = widgets.Output()
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    out_pl
    ```
      * grabbing image
      * using `Output` widget to display the image uploaded through the widget
    * `pred, pred_idx, probs = learn_inf.predict(out_pl)`
      * getting predictions for the image
    ```
    lbl_pred = widgets.Label()
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    lbl_pred
    ```
      * using `Label` to display the predictions
    ```
    btn_run = widgets.Button(description='Classify')
    btn_run
    ```
      * button for classification
    ```
    def on_click_classify(change):
        img = PILImage.create(btn_upload.data[-1])
        out_pl.clear_output()
        with out_pl: display(img.to_thumb(128,128))
        pred,pred_idx,probs = learn_inf.predict(img)
        lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

    btn_run.on_click(on_click_classify)
    ```
      * click even handler
      * function to be called when button is pressed
    ```
    btn_upload = widgets.FileUpload()
    VBox([widgets.Label('Select your bear!'), btn_upload, btn_run, out_pl, lbl_pred])
    ```
      * finished GUI
  * Converting Notebook into App
    ```
    !pip install voila
    !jupyter serverextension enable voila —sys-prefix
    ```
      * installing Voilà
      * cells beginning with a `!` contain code that is passed to the shell in notebook
      * replace "notebooks" in browsers URL with "voila/render" to view notebook as a Voilà web app
      * can use `(pred,pred_idx,probs = learn.predict(img))` with any framework
  * Deploying an app
    * GPU is needed to train deep learning models, but is not needed to serve the model in production
      * GPUs are only useful when doing lots of identical work in parallel
      * image classification is usually one task (classification) at a time
      * CPU can be more cost-effective
    * "You may well want to deploy your application onto mobile devices, or edge devices such as a Raspberry Pi. There are a lot of libraries and frameworks that allow you to integrate a model directly into a mobile application. However, these approaches tend to require a lot of extra steps and boilerplate, and do not always support all the PyTorch and fastai layers that your model might use [...] we recommend wherever possible that you deploy the model itself to a server, and have your mobile or edge application connect to it as a web service. "
* common issues
  * out of domain data
    * data that model sees in production which is different to what was seen in training
  * domain shift
    * when the type of data our model sees changes over time
  * models may change the behavior of the system they are apart of
    * feedback loops can result in negative implications of bias getting worse and worse
    
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
  * class that stores validation and training `DataLoader` objects
* What four things do we need to tell fastai to create DataLoaders?
  * what kinds of data we are working with
  * how to get the list of items
  * how to label these items   
  * how to create the validation set
* What does the splitter parameter to DataBlock do?
  * divides the training and validation sets
* How do we ensure a random split always gives the same validation set?
  * setting the seed (`RandomSplitter(valid_pct=0.2, seed=42)`)
* What letters are often used to signify the independent and dependent variables?
  * x and y, respectively
* What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
  * crop: crops the images to fit a square shape of the size requested using the full width or height
  * pad: pads the images with 0 to fit the size, resulting in a lot of empty space
  * squish: squish or stretch images resulting in unrealistic shapes
  * crop tends to be the best since it reflects how images work in the real world
* What is data augmentation? Why is it needed?
  * generating variations of input to expand a dataset and prevent overfitting
* What is the difference between item_tfms and batch_tfms?
  * `item_tfms` parameter which tells how to transform a single item and `batch_tfms` tells fastai to perform the transformations over the entire batch
* What is a confusion matrix?
  * visualization of the actual categories of the dataset vs the predicted categories of the items
* What does export save?
  * the model architecture and trained parameters
* What is it called when we use a model for getting predictions, instead of training?
  * inference
* What are IPython widgets?
  *  GUI components with JavaScript and Python functionality in a web browser
* When might you want to use CPU for deployment? When might GPU be better?
  * CPU is usually better for deployment as it is more cost-effective than GPUs.
  * GPU might be better if deployment is done in large batches
* What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
  * application will require a network connection and there will inevitably be latency
* What are three examples of problems that could occur when rolling out a bear warning system in practice?
  * inability to handle nighttime images
  * touble dealing with low-resolution camera images
  * working with video data instead of images
* What is "out-of-domain data"?
  * data that model sees in production which is different to what was seen in training
* What is "domain shift"?
  * when the type of data our model sees changes over time
* What are the three steps in the deployment process?
  * Manual Process
  * Limited scope deployment
  * gradual expansion
