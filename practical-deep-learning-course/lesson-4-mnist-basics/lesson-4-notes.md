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
        - the mean of the absolute value of differences
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
   - should check shapes as we go
 - functions to calculate mean absolute error between two tensors
   ```
   def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
   a_3  # tensor representing a single image
   mean3  # tensor representing the ideal image
   mnist_distance(a_3, mean3)  # tensor(0.1114)
   ```
   - would need to calculate distance from ideal image for every image
 - we can pass in tensor representing all images in our validation set and compare them simultaneously with the ideal image
   ```
   valid_3_tens  # the tensor representing all 3s in the validation set
   mean3  # the ideal 3 image
   valid_3_dist = mnist_distance(valid_3_tens, mean3)  # passing in all 3s as argument to our distance function
   valid_3_dist  # tensor([0.1050, 0.1526, 0.1186,  ..., 0.1122, 0.1170, 0.1086]), the distance for each image
   valid_3_dist.shape  # torch.Size([1010]), confirming we have computed the distance for all 1010 images
   ```
 - broadcasting
   - PyTorch will recognize when two tensors are of different ranks as it tries to perform an operation between them
   - it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank
   - after broadcasting so two argument tensors have the same rank, PyTorch performs the operation on each corresponding element of the two tensors, and returns the tensor result
   - PyTorch treats `mean3`, a rank-2 tensor representing a single image, as if it were 1010 copies of the same image, then subtracts each of those copies from each 3 in our validation set
     ```
     (valid_3_tens-mean3).shape  # torch.Size([1010, 28, 28])
     ```
   - calculating the difference between our "ideal 3" and each of the 1010 3s in the validation set, for each 28x28 image, resulting in shape `[1010, 28, 28]`
   - PyTorch doesn't actually copy `mean3` 1010 (doesn't actually allocate any additional memory) but rather pretends it were a tensor of that shape
   - PyTorch does the whole calculation in C
     - if using a GPU, calculates in CUDA, the equivalent of C in GPU
     - these calculations in C are tens of thousands of times faster than pure Python
     - these calculations on a GPU are up to a million times faster than pure Python
   - all broadcasting and elementwise operations and functions done in PyTorch act this way
   - `abs` applies method to each individual element in the tensor and returns a tensor of the result
   - `mean((-1,-2))`
     - `(-1,-2)` tuple represents a range of axes
     - `-1` refers to last element and `-2` refers to second to last element
     - we are taking the mean ranging over the values indexed by the last two axes of the tensor
     - the last two axes are the horizontal and vertical dimensions of the image, so we are left with just the first tensor axis, which indexes over our images which is why our final size is `1010`
     - for every image, we averaged the intensity of all the pixels in that image
 - deciding whether an image is a 3 or not
   - if the distance between the digit in question and the ideal 3 is less than the distance to the ideal 7, then its a 3
     ```
     def is_3(x): return mnist_distance(x,mean3) < mnist_distance(x,mean7)
     is_3(a_3), is_3(a_3).float()  # (tensor(True), tensor(1.))
     is_3(valid_3_tens)  # tensor([True, True, ..., True])
     ```
   - calculating the accuracy for each of the 3s and 7s
     ```
     accuracy_3s = is_3(valid_3_tens).float().mean()
     accuracy_7s = (1-is_3(valid_7_tens).float()).mean()  # inverse of is_3() to check if image is 7
     accuracy_3s  # tensor(0.9168)
     accuracy_7s  # tensor(0.9854)
     (accuracy_3s+accuracy_7s)/2  # tensor(0.9511), average accuracy
     ```
   - this is our baseline, now we must train our actual model

### Stochastic Gradient Descent (SGD)
- instead of trying to find the similarity between an image and an 'ideal image' we could instead look at each individual pixel and come up with a set of weights for each one
  - the highest weights would be associated with pixels most likely to be black for a particular category
  - pixels in the bottom right are not likely to be activated for a 7, so they should have a low  weight for a 7
  - pixels in the bottom right are more likely to be activated for an 8, so they should have high weight for an 8
- can create a function that takes into account an image and the wieghts for each category
  ```
  def pr_eight(x,w) = (x*w).sum()
  ```
  - we assume `x` is an image representing a vector
  - we assume `w` is a vector of all the weights
    - we want 'w' such that the result of our function is high for actual 8s and low for images of other numbers
- steps to turn function into machine learning classifier:
  1. Initialize the weights
  2. For each image, use the weights to predict whether it appears to be a 3 or a 7
  3. Based on predictions, calculate how good the model is (its loss)
  4. Calculate the gradient, which measures for each weight, how changing that weight would change the loss
  5. Change all the weights based on the gradient
  6. Back to step 2 and repeat the process
  7. Iterate until you decide to stop the training process (when the model is good enough)
- guidelines
  - initialize
    - we initialize parameters to random values
  - loss
    - we need a function that returns a number that is small if the performance of the model is good
    - conventionally, we treat a small loss as good, and a large loss as bad
  - step
    - we use calculus to calculate the gradients which help us to determine in which direction and by how much to change the weights
  - stop
    - we train until the accuracy of the model starts getting worse or until we reach a specified number of epochs

#### Calculating Gradients
- PyTorch computes the derivative of nearly any function for us
  ```
  xt = tensor(3.).requires_grad_()
  ```
  - `requires_grad_` tells PyTorch we want to calculate the gradients with respect to that variable at that value
  - we 'tag' the variable so PyTorch will remember to keep track of how to compute gradients of the other, direct calculations on it that we will ask for
  - in math the 'gradient' of a function is another function
  - in deep learning 'gradients' refer to the value of a function's derivative at a particular argument value
    - PyTorch API puts the focus on the argument rather than the function we're actually computing the gradients of
  - when we calculate our function with a value tagged for gradient, a gradient function is returned along with a value
    ```
    yt = f(xt)  # where f(p) = p**2
    yt  # tensor(9., grad_fn=<PowBackward0>)
    ```
  - telling PyTorch to calculate the gradients
    ```
    yt.backward()
    ```
    - `backward` refers to backpropagation, the process of calculating the derivative of each layer
  - we view the gradients by checking the `grad` attribute of the original tensor
    ```
    xt.grad  # tensor(6.)
    ```
    - sure enough derivative of `x**2` is `2*x` and evaluated at `x=3` we get `6`
  - we can do the same passing a vector as argument
    ```
    xt = tensor([3.,4., 10.]).requires_grad_()
    xt  # tensor([3., 4., 10.], requires_grad=True)
    def f(x): return (x**2).sum()  # f: rank 1 tensor (vector) -> rank 0 tensor (scalar)
    yt = f(xt)
    yt  # tensor(125., grad_fn=<SumBackward0>)
    yt.backward()
    xt.grad  # tensor([6., 8., 20.])
    ```
  - note: the gradients only give us the slope of our function, not how far to adjust the parameters

#### Stepping with a Learning Rate
- the learning rate (LR)
  - some small number which we multiply by our gradient and the product of which we use to change our parameters
  - usually a number between 0.001 and 0.1
  - after picking a learning rate, the parameters are adjusted: `w -= gradient(w) * lr`
    - this is *stepping* our parameters, using an *optimizer step*
  - picking a learning rate that is too low can mean having to do a lot of steps
  - picking a learning rate that is too high can result in the loss actually getting worse
    - may also just 'bounce' around rather than diverging

#### SGD Example
- 

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
5. **What is a "rank-3 tensor"?**
  - a three dimensional tensor
6. **What is the difference between tensor rank and shape? How do you get the rank from the shape?**
  - tensor rank is the number of axes or dimensions of a tensor
  - tensor shape is the size of each axis of a tensor
  - `rank = len(tensor.shape)`
7. **What are RMSE and L1 norm?**
  - L1 norm is the mean of the absolute diffrence between all values in two vectors
  - L2 norm or RMSE is the square root of the mean of the square of differences of all values in two vectors
8. **How can you apply a calculation on thousands of numbers at once, many thousands of times faster than a Python loop?**
  - by using NumPy or PyTorch operations
9. **Create a 3×3 tensor or array containing the numbers from 1 to 9. Double it. Select the bottom-right four numbers.**
  ```
  t = tensor([[o,o+1,o+2] for o in range(1,10,3)])
  t = t*2
  bottom_right = t[1:,1:]
  ```
10. **What is broadcasting?**
  - A method that PyTorch uses when performing elementwise operations on two tensors of different rank
  - PyTorch expands the tensor of the smaller rank to have the same size as the tensor of the larger rank (or acts as if it does so without actually allocating any extra memory), and then performs the elementwise operations on the two tensors
11. **Are metrics generally calculated using the training set, or the validation set? Why?**
  - metrics are calculated using the validation set in order to prevent overfitting, so that we don't build a model that only memorizes the training data
12. **What is SGD?**
  - a method for automatically updating the weights of our model based on a loss function
13. **Why does SGD use mini-batches?**
  - 
14. **What are the seven steps in SGD for machine learning?**
15. **How do we initialize the weights in a model?**
  - randomly
16. **What is "loss"?**
  - a measure of the performance of our model
  - usually, a small loss is treated as good and a large loss is treated as bad
17. **Why can't we always use a high learning rate?**
  - 
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
