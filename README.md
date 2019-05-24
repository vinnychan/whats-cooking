# Whats Cooking

Using RNN to Predict the Cuisine given the ingredients of a dish

Steps we took:
1. Formatting the data into matrices
2. Augmenting the data (adding more data with different permutations of the ingredients)
3. Train the model using RNN on multiple data points in the training data set
4. Test model by predicting on sample training data


### How to Run

Run `python train.py` to train the network

Run `python predict.py ['array', 'of', 'ingredients']`

![prediction](https://i.imgur.com/x0sIBiI.png)

Run `python server.py` to run locally as a server for a web API to predict

### Results

These results only reflect the accuracy on 10,000 random training pairs during the training.

Original data set (~12MB): 51.43% (5th generation)\
Augmented data set (~68MB): 58.06% (1st generation)

The augmented data includes 10 permutations of ingredient sets.

Example:
```
greek: [
    ["tomato", "lettuce", "cucumber],
    ["lettuce", "cucumber, "tomato"], // permutated set
    ...
    ["orange", "apple"]
]
```
