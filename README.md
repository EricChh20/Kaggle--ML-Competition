# Machine Learning - Kaggle Competitions 

**CS460-Machine Leaning in-class competitions**

### Prerequisites: 
```
Python: 3.6 
Scikit-Learn: 0.22 
Keras: 2.2.5
```



## Competetion 1: Tiny Images 
- Link: https://www.kaggle.com/c/ukycs460g2019p1 

### CNN with transfer learning: Keras 
- Attempted a CNN with transfer learning and data augmentation but got poor results (obviously) with the dataset. 
- Attempted base models of MobileNet and InceptionV3 with ImageNet weights while fine-tuning the model with augmented data. 
- Achieved good training accuracy but really poor testing results. Perhaps I could have tried more regularization to avoid the overfitting but due to such small datasets, I just went to simpler methods. 

### Random Forest: 
- Hyperparameters: n_estimators=3000, max_depth=15, max_features=15
- Utilized the built-in grid search function to find the most optimal set of parameters. The range of values included: 
- n_estimators: [500, 1000, 1500, 2000, 2500, 3000]
- max_depth: [1, 2, 5, 7, 9, 15]
- max_features: [5, 15, 25, 50, 100, 150, 200] 


## Competetion 2: Pixel Prediction
- Link https://www.kaggle.com/c/roxie/

### Multi-layered Perceptron Regressor 
- Hyperparameters: hidden_layer_sizes=(100,50,50,20), solver=adam, activation=tanh. 
- Utilized the built-in grid search function to find the most optimal set of parameters. The range of values included: 
- Hidden_layer_sizes: [(100), (100,50,50,20)]
- Activation: [tanh, relu] 
- Solver: [sgd, adam]


## Competetion 3: Bird Mapping
- https://www.kaggle.com/c/what-bird-would-i-see

### MLP Classifier
- Hyperparameters: hidden_layer_sizes = (100, 100, 50, 20), activation = tanh, solver = adam. 
- Utilized the built-in grid search function to find the most optimal set of parameters. The range of values included: 
- Hidden_layer_sizes = [(100, 100, 50, 20), (50,50,20), (200, 100, 100, 50)]
- Activation = [tanh, relu]
- Solver = [sgd, adam]
- I experimented a bit with the training features and seem to get the best results from using the location features and the month. 


## Competetion 4: Sinkhole or not
- https://www.kaggle.com/c/sinkhole-or-not-2019/

