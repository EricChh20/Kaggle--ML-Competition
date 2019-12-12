# Kaggle-ML-Competition

# Machine Learning - Kaggle Competitions 

**CS460-Machine Leaning in-class competitions**


### Competetion 1: Tiny Images 
- Link: https://www.kaggle.com/c/ukycs460g2019p1 

- CNN with transfer learning: Keras 
o	Attempted a CNN with transfer learning and data augmentation but got poor results (obviously) with the dataset. 
o	Attempted base models of MobileNet and InceptionV3 with ImageNet weights while fine-tuning the model with augmented data. 
o	Achieved good training accuracy but really poor testing results. Perhaps I could have tried more regularization to avoid the overfitting but due to such small datasets, I just went to simpler methods. 

-	Random Forest: 
o	Hyperparameters: n_estimators=3000, max_depth=15, max_features=15
o	Utilized the built-in grid search function to find the most optimal set of parameters. The range of values included: 
	n_estimators: [500, 1000, 1500, 2000, 2500, 3000]
	max_depth: [1, 2, 5, 7, 9, 15]
	max_features: [5, 15, 25, 50, 100, 150, 200] 


### Competetion 2: Pixel Prediction
- Link https://www.kaggle.com/c/roxie/

### Competetion 3: Bird Mapping
- https://www.kaggle.com/c/what-bird-would-i-see

### Competetion 4: Sinkhole or not
- https://www.kaggle.com/c/sinkhole-or-not-2019/

