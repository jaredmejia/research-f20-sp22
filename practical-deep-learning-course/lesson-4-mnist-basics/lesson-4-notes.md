# Notes from Lesson 4: Training a Digit Classifier
## Image Classifier on MNIST Data Set
- the MNIST dataset
  - images of handwritten digits collected/turned into dataset Yann Lecun and co
- `Image` class from the Python Imagining Library (PIL)
  - PIL is most widely used Python package for opening, manipulating, and viewing images
  - Jupyter works well with PIL and allows us to display images automatically
  - ex: `Image.open(path)`
- numbers that make an image
  - must convert numbers to *NumPy array* or *PyTorch tensor*
  - part of image as NumPy Array
    - `array(image3)[4:10,4:10]`
    - NumPy indexes from top to bottom and left to right
    - code shows section located in top left
  - part of image as PyTorch tensor
    - `tensor(image3)[4:10,4:10]`
  - pixels of mnist numbers
    - background white pixels stored as 0
    - black is the number 255
    - shades of grey between 0 and 255
- baseline
  - simple model you are confident will perform reasonably well
  - should be simple to implement, easy to test
  - allows you to test each of improved, more complicate methods to ensure they do better than baseline
  - could use easy to implement model or use others models to compare against
- Creating baseline for MNIST
  - finding the average pixel value for every pixel of 3s and doing the same for 7s
    - creating a list of all image tensors for 3 and a list of all image tensors for 7
      ```
      seven_tensors = [tensor(Image.open(o)) for o in sevens]
      three_tensors = [tensor(Image.open(o)) for o in threes]
      len(three_tensors), len(seven_tensors)
      ```
    - displaying image of a tensor with fastai `show_image` function
      ```
        show_image(three_tensors[1])
      ```
    - want to compute average over all images of the intensity of a pixel
      - need to combine all images in list into three-dimensional tensor
      - this is a *rank-3 tensor*
      - PyTorch `stack` allows us to stack up individual tensors in a collection into a single tensor
      - stacking up 3 and 7 individual tensors into two respective tensors
        ```
        stacked_sevens = torch.stack(seven_tensors).float()/255
        stacked_threes = torch.stack(three_tensors).float()/255
        ```
      - we need to cast to float
      - when images are float pixel values are expected to be between 0 and 1 so we divide by 255
      - *shape* of a tensor tells you the length of each axis
        ```
        stacked_threes.shape  # torch.Size([6131, 28, 28])
        ```
      - the length of a tensor's shape is its *rank* 
        ```
        len(stacked_threes.shape)  # 3
        stacked_threes.ndim  # 3
        ```
      - we calculate the mean of all image tensors by taking the mean along the dimension that indexes over all the images
      - for every pixel position, computing the average of that pixel over all the images
        ```
        mean3 = stacked_threes.mean(0). # 0 here is the dimension that indexes over all images
        mean3.shape  # torch.Size([28, 28])
        ```
      - this results in a value for every pixel position, a single (blurry) image
    - finding the distance of a single 3 from the 'ideal' 3
      - the *L1 norm*
        - the mean of the absolute value of idfferences
        ```
        dist_3_abs = (a_3 - mean3).abs().mean()
        dist_7_abs = (a_3 - mean7).abs().mean()  # if greater than dist_3_abs, prediction is 3
        ```
      - the *L2 norm* or *root mean squared error* (RMSE)
        - the square root of the mean of the square of differences
        ```
        dist_3_sqr = ((a_3-mean3)**2).mean().sqrt()
        dist_7_sqr = ((a_3-mean7)**2).mean().sqrt(). # if greater than dist_3_sqr, prediction is 3
        ```
      - Pytorch *loss functions*
        - `torch.nn.functional' package
          ```
          import torch.nn.functional as F
          F.l1_loss(a_3.float(),mean7)  # the l1 loss or mean absolute value or L1 norm
          F.mse_loss(a_3,mean7).sqrt(). # the mean squared error
          ```
        - L1 norm vs MSE
          - MSE penalizes bigger mistakes more heavily
          - MSE also more lenient with small mistakes
### NumPy Arrays and PyTorch Tensors
  - NumPy
    - most widely used library for scientific and numeric programming in Python
    - provides similar funcitonality and API to that of PyTorch
    - unlike PyTorch, does not support using the GPU or calculating gradients, which are both critical for deep learning
      - Pytorch > NumPy for deep learning
  - Python slow compared to many languages
    - anything fast in Python, NumPy, or PyTorch usually a wrapper for a compiled object written (and optimized) in another language
      - specifically C
    - NumPy arrays and PyTorch tensors can finish computations many thousands of times faster than using pure Python
  - NumPy array
    - multidimensional table of data with all items of the same type
      - even beyond dimension 3
    - items can be arrays of arrays (innermost potentially of different sizes, a 'jagged array')
    - if items are all of some simple type (i.e. integer or float), NumPy stores them as a compact C data structure in memory
      - NumPy has wide variety of operators and methods that can run computations on these compact structures at the same speed as optimized C, since they are written in optimized C
  - PyTorch tensor
    - similar to NumPy array, but with an additional restriction
    - multidimensional table of data, with all items of *the same* type
      - the type must be a single basic numeric type for all components
      - a PyTorch tensor cannot be jagged
      - a PyTorch tensor is always a regularly shaped multidimensional rectangular structure
    - additional capabilities
      - Pytorch tensors can live on the GPU
        - their computation is optimized for the GPU and can run much faster (given lots of values to work on)
      - PyTorch can automatically calculate derivatives of these operations including combinations of operations
  - to take advantage of the speed of C while programming in python, avoid writing loops and replacing them with commands that work directly on arrays or tensors
  - using the array/tensor APIs
    - to create an array or tensor, pass a list (or list of lists of...) to `array()` or `tensor()`
      ```
      data = [[1,2,3],[4,5,6]]
      arr = array(data)  # array([[1,2,3],[4,5,6]])
      tns = tensor(data)  # tensor([[1,2,3],[4,5,6]])
      ```  
    - selecting a row (arrays and tensors are 0-indexed)
      ```
      arr[1]
      tns[1]  # tensor([4,5,6])
      ```
    - selecting a column (using `:` to indicate all of the first axis)
      ```
      arr[:,1]
      tns[:,1]  # tensor([2,5])
      ```
    - part of a row/col
      ```
      arr[:,1:3]
      tns[:,1:3]  # tensor([[2,3],[5,6]])
      ```
    - can use standard operators such as `+`, `-`, `*`, `/` to perform operations on all items
    - tensors have a type
      - `torch.FloatTensor`, `torch.LongTensor`, etc.
      
### Computing Metrics Using Broadcasting
- metric is calculated based on predictions of our model and the correct labels in our dataset in order to tell us how good our model is
- in practice *accuracy* is the metric for classification models
- metric is calculated over a *validation set* in order to prevent overfitting
 - creating tensors to be used for calculating a metric measuring the quality of our inital model which measures distance from an ideal image
   ```
   valid_3_tens = torch.stack([tensor(Image.open(o))
                               for o in (path/'valid'/'3').ls()])  # creating tensors for 3s in validation directory
   valid_3_tens = valid_3_tens.float()/255
   valid_3_tens.shape(). # torch.Size([1010, 28, 28]), checking shapes as we go
   ```
 
    

## Questionnaire
1. **How is a grayscale image represented on a computer? How about a color image?**
  - grayscale image: each pixel is number 0 (white) to 255 (black)
  - color image: each pixel is three values 0 to 255
2. **How are the files and folders in the MNIST_SAMPLE dataset structured? Why?**
  - different folders for the training set, validation set, and the test set
  - training set only includes folders for 3 and 7 in order to help the model to learn to generalize when it is training
3. **Explain how the "pixel similarity" approach to classifying digits works.**
  - first finding the average pixel value for every pixel
  - second seeing which of the two ideal digits a given image is most similar to
4. **What is a list comprehension? Create one now that selects odd numbers from a list and doubles them.**
  - python way to create a list faster than with a loop
  - includes a collection that is being iterated over, an optional filter, and a function for each element
    ```
    divisible_by_47 = [o for o in range(10000) if o % 47 == 0]
    ```
5. What is a "rank-3 tensor"?
  - a three dimensional tensor
6. What is the difference between tensor rank and shape? How do you get the rank from the shape?
  - tensor rank is the number of axes or dimensions of a tensor
  - tensor shape is the size of each axis of a tensor
  - `rank = len(tensor.shape)`
7. What are RMSE and L1 norm?
  - L1 norm is the mean of the absolute diffrence between all values in two vectors
  - L2 norm or RMSE is the square root of the mean of the square of differences of all values in two vectors
8. How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?
  - by using NumPy or PyTorch operations
9. Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.
  ```
  t = tensor([[o,o+1,o+2] for o in range(1,10,3)])
  t = t*2
  bottom_right = t[1:,1:]
  ```
10. What is broadcasting?
11. Are metrics generally calculated using the training set, or the validation set? Why?
12. What is SGD?
13. Why does SGD use mini-batches?
14. What are the seven steps in SGD for machine learning?
15. How do we initialize the weights in a model?
16. What is "loss"?
17. Why can't we always use a high learning rate?
18. What is a "gradient"?
19. Do you need to know how to calculate gradients yourself?
20. Why can't we use accuracy as a loss function?
21. Draw the sigmoid function. What is special about its shape?
22. What is the difference between a loss function and a metric?
23. What is the function to calculate new weights using a learning rate?
24. What does the DataLoader class do?
25. Write pseudocode showing the basic steps taken in each epoch for SGD.
26. Create a function that, if passed two arguments [1,2,3,4] and 'abcd', returns [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]. What is special about that output data structure?
27. What does view do in PyTorch?
28. What are the "bias" parameters in a neural network? Why do we need them?
29. What does the @ operator do in Python?
23. What does the backward method do?
24. Why do we have to zero the gradients?
25. What information do we have to pass to Learner?
26. Show Python or pseudocode for the basic steps of a training loop.
27. What is "ReLU"? Draw a plot of it for values from -2 to +2.
28. What is an "activation function"?
29. What's the difference between F.relu and nn.ReLU?
30. The universal approximation theorem shows that any function can be approximated as closely as needed using just one nonlinearity. So why do we normally use more?
