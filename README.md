# 📌 Project Background  
This project aims to develop an automated system for classifying fruits as fresh or rotten using deep learning techniques. With over 4000 images in the dataset, the system will leverage convolutional neural networks to analyze visual features and determine fruit freshness. This approach addresses the challenge of efficient quality control in the food industry, potentially reducing food waste and improving consumer satisfaction. By automating the classification process, the project seeks to overcome the limitations of time-consuming and subjective manual inspections, offering a more consistent and scalable solution for fruit quality assessment.
[dataset]([https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification/data)).

<i>In collaboration with Vincent Lee, Wenchi Tseng, Wei Chen, I-Ching Lu, Zihang Sun.</i>

# 🧐 Data Overview  
This dataset contains 4000+ images, each in one of the following categories:  
- apple fresh
- apple rotten
- banana fresh
- banana rotten

The outcome variable is binary (1 for fresh and 0 for rotten). Images in the "eye opened" or "person not yawning" groups are considered non-drowsy, whereas images in the "eye closed" or "person yawning" groups are considered drowsy. 50% of the images in this dataset have an outcome of 1, or fresh.

# 🖥️ Data Processing  
To pre-process the data, we did data partitioning, image processing, and normalization. 
- Data partitioning to split the images into three datasets, 60% in training, 20% in cross-validation, and the remaining 20% in testing. This resulted in 1740 images in the training set and 580 images in each of the cross-validation and testing sets.  
  <br>
    <img src="Images/img-02.png" width="800">
  <br>   
- Image processing to resize each of the images to 64 pixels x 64 pixels and to add the 0 and 1 labels to the images.  
  <br>
    <img src="Images/img-01.png" width="800">
  <br>  
- Normalizing the images by scaling the features to be between 0 and 1, making the data more interpretable.  
  <br>
    <img src="Images/img-03.png" width="800">
  <br>  

# 📈 Evaluation of Models  
The evaluation metric that we used was accuracy. Not only is accuracy easy to explain, but overall model correctness is important. Misclassifying a drowsy driver as non-drowsy can be dangerous, and misclassifying alert drivers as drowsy could lead to unnecessary alerts or make the system less robust. In general, humans can identify whether someone is yawning or an image of an eye is open or closed, so the human error is close to 0% meaning that our goal should be to try to aim for high accuracy. Our benchmark model, logistic regression, returned a training accuracy of 90.7%.  

We then tried building two simple neural networks, one with 1 hidden layer (4 nodes) and another one with 2 hidden layers (7 nodes in hidden layer 1 and 4 nodes in hidden layer 2). The neural network with 1 hidden layer returned the best results with a training accuracy of 99%, a cross-validation accuracy of 98.5%, and a testing accuracy of 97%.  

Though the performance of the neural network with 1 hidden layer was extremely positive, we used the Tensorflow package to try to run the dataset through different models to see if they would provide better results. Some algorithms and techniques we tried included:   
- Adam (Adaptive Moment Estimation)
- Mini-Batch Gradient Descent
- Mini-Bath Gradient Descent with Momentum
- RMSprop (Root Mean Squared Prop)
- L2 Regularization
- Dropout
- Batch Normalization
- Early Stopping
- A combination of the algorithms and techniques

However, the best model was still the neural network model with 1 hidden layer and 4 nodes.  

# ❕Project Importance  
Industries that are or could be impacted by a driver drowsiness detection system include car manufacturers, transportation services, and businesses that use heavy machinery. 

Why is a neural network model more useful than a simple model, like logistic regression, for this dataset?  
When detecting drowsiness, image data is more accurate than variable data and neural networks are better at image classification than simple models are. This is one of the reasons why the model accuracy on the neural network is better than the logistic regression accuracy.  

Who would benefit from a more accurate driver drowsiness detection model?  
Financially, auto insurance companies, tech companies that will or are developing these systems, and auto manufacturers will benefit from this highly accurate model. And more importantly, the general public will benefit from this in terms of greater traffic safety.  

# ☁️ Project Improvements  
Models and percentages are a good starting point. However, to make this project more accessible to others, an improvement may be to put this into a friendly user interface.  
