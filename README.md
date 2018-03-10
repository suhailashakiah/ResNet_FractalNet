# ResNet Trained on CIFAR 10 with 91% accuracy
A ResNet with 20 layers was built and trained on 1 Tesla K80 GPU and 4 CPUs.

#### Code Heirarchi:
##### 1. loading_data.ipynb

   Loads the downloaded data, preprocesses it and stores it in a numpy array for feeding into the neural network. The CIFAR 10 dataset can be downloaded [here](https://www.cs.toronto.edu/~kriz/cifar.html). 
   

##### 2. resnet_training_testing.ipynb

   Contains building model, training, and testing. All training information is provided in this notebook.


##### 3. The fully trained model architecture and weights are provided. 

   The model architecture is provided in the file 'model_epoch127_json.pkl'.

   The trained weights are provided in 'model_epoch127_weights.h5'
   
   The model can be reconstructed by using the following lines of code in keras after importing the required libraries:
   
   
```
   import keras
   import pickle 
   from keras.models import load_model
   from keras.models import model_from_json

   json_string = pickle.load( open( "model_epoch127_json.pkl", "rb" ) )
   model = model_from_json(json_string)
   model.load_weights('model_epoch127_weights.h5')
```
   
# Results 
The loss history during training is shown in the figure below: 

![loss_history](https://user-images.githubusercontent.com/18056877/37247169-528a648a-2485-11e8-9314-7a57829586ab.png)

The training accuracy:

![training_accuracy](https://user-images.githubusercontent.com/18056877/37247175-6e7c5f90-2485-11e8-8625-20d30b260d9f.png)

Testing accuracy:

![testing_accuracy](https://user-images.githubusercontent.com/18056877/37247178-77daca04-2485-11e8-8a3e-68364a027be6.png)

# Model Architecture
The original resnet paper can be obtained [here](https://arxiv.org/abs/1512.03385).
The model architecture is shown below: 

![resnetv1_model](https://user-images.githubusercontent.com/18056877/37247163-194b92f2-2485-11e8-9a3d-2732ef511976.png)
