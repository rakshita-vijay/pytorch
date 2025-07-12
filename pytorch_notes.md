torch.tensor: multi-dimensional matrix containing elements of the same data type

torch.nn module: contains tools for building neural networks

torch.nn.functional: has functions like activation functions and loss functions

class Model(nn.Module) 
    nn.Module is the base class for all neural network models in PyTorch. 
    By inheriting from it, your class gets all the features needed for a neural network.  

self.fc1 = nn.Linear(in_features, h1):
    Creates the first fully connected (linear) layer. 
    Takes in_features inputs and outputs h1 values (neurons).

relu: rectified linear unit
      common activation f() used in nn, especially deep learning models
      helps the network learn complex patterns by introducing non-linearity (model can learn more than just straight lines or simple relationships)
 
f(x) = max(0, x) 
- If x > 0, output x
- If x <= 0, output 0 

---

# Step-by-Step Guide: How to Trained a Model
 
## 1. **Import Libraries**
You brought in the tools needed for the job:
- **PyTorch** for building and training the model.
- **Pandas** for handling data.
- **Matplotlib** for making graphs.
- **Scikit-learn** for splitting data into training and test sets. 

## 2. **Get and Prepare the Data**
- Downloaded the dataset.
- Converted string data to numbers (so the computer can work with them).
- Split the data into features (inputs) and labels (outputs).

## 3. **Convert Data for PyTorch**
- Changed the data from a DataFrame to numpy arrays (for easier handling).
- Split the data into training and test sets (using 80% for training, 20% for testing).
- Converted the arrays to PyTorch tensors (the format PyTorch understands). 

## 4. **Build the Neural Network Model**
- **Created a class for the model** with:
  - An input layer (to take in the training/test/prediction data).
  - Two/X hidden layers (to help the model learn patterns).
  - An output layer (to predict).

*Why?*  
A neural network learns by passing data through these layers, adjusting itself to get better at making predictions.

## 5. **Set the Random Seed**
- Set a manual seed for randomness - to ensure you get the same results every time you run the code, which is important for checking and comparing your work.

## 6. **Set Up the Loss Function and Optimizer**
- **Loss function:** Used CrossEntropyLoss to measure how wrong the model’s guesses are.
- **Optimizer:** Used Adam to adjust the model’s weights and biases to reduce errors.
- **Set a learning rate** (how big the model’s steps are when learning).

*Why?*  
The loss function tells the model how well it’s doing, and the optimizer helps it get better by adjusting its internal numbers.

## 7. **Train the Model**
- **Chose a number of epochs** (how many iterations - number of times it has to go through the data).
- **For each epoch:**
  - Made predictions on the training data.
  - Measured the loss (error).
  - Used backpropagation to adjust the model.
  - Recorded the loss to see how it improves.

*Why?*  
Training over many epochs helps the model learn the patterns in the data. Tracking the loss shows if it’s learning.

## 8. **Visualize the Training**
- Plotted a graph of loss vs. epochs - to see if the model’s error is going down, which means it’s learning.

## 9. **Evaluate the Model**
- **Tested the model on the test set** (data it never saw during training).
- **Measured the test loss** to see how well it performs on new data.
- **Checked predictions one by one** to see if they matched the real answers.
- **Counted how many were correct and wrong**.

## Summary Table

| Step                        | What You Did                                 | Why You Did It                                      |
|-----------------------------|----------------------------------------------|-----------------------------------------------------|
| Import Libraries            | Loaded PyTorch, Pandas, etc.                 | Tools for data, modeling, and visualization         |
| Get and Prepare Data        | Downloaded, cleaned, split data              | Model needs numbers and clear inputs/outputs        |
| Convert Data for PyTorch    | Changed to arrays/tensors, split sets        | PyTorch needs tensors; need test data for checking  |
| Build the Model             | Defined layers in a class                    | Neural network structure for learning               |
| Set the Random Seed         | Fixed the randomness                         | Makes results repeatable                            |
| Set Loss & Optimizer        | Chose error function and optimizer           | Guides and improves learning                        |
| Train the Model             | Ran many epochs, tracked loss                | Model learns from data, gets better over time       |
| Visualize Training          | Plotted loss curve                           | See if learning is working                          |
| Evaluate the Model          | Tested on new data, checked accuracy         | See if it works on unseen examples                  |
