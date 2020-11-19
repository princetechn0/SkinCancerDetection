# Skin Cancer Detection


## Skin Cancer Neural Network Detection using Tensorflow and CNN Learning


### Based on DataSet found here: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/kernels


## Abstract
Skin cancer holds the unfortunate distinction of being the world’s most common cancer. It is also commonly misdiagnosed, which is problematic if the cancer spreads. Therefore, detecting its presence early is crucial in ensuring the safety of affected individuals. By using a neural network suited to detect the most common forms of skin cancer, misdiagnosis can be drastically reduced. According to SkinCancer.org, skin cancer can be misdiagnosed up to 40 percent of the time, for example, in younger patients with cancerous skin lesions. 

## Introduction
Skin cancer is the most common form of cancer. Current estimates are that one in five Americans will develop skin cancer in their lifetime. It is estimated that 192,310 new cases of melanoma will be diagnosed in the U.S. in 2019. Therefore, properly diagnosing a skin lesion as a cancer is important in protecting the lives of those affected. 

Diagnosis of skin cancers is a process that requires medical professionals, extraction of a sample, and a plethora of other steps in order to reach a conclusion. Furthermore, achieving a valid diagnosis is also an issue. Skin cancer is increasingly misdiagnosed by physicians. In many cases, skin cancer is incorrectly diagnosed as eczema or another less serious disease. Misdiagnoses, failure to diagnose, and delayed diagnosis can all be very dangerous for the patient, as the cancer continues to progress without treatment.

By developing a neural network capable of processing image samples of the most common forms of skin cancers, the likeliness of a misdiagnosis can be dramatically reduced. A clinical visit is still required in order to obtain a sample, however, the event of a misdiagnosis occurring should be decreased. The two approaches employed both utilize a convolutional neural network, except one of them uses transfer learning in order to improve the accuracy of the prediction.  

## Problem Formulation
Detecting skin cancers from samples of lesions requires the use of an image sample processed by a medical professional, meaning the patient still needs to visit the doctor. The image sample is then processed by the neural network, which transforms it into an array of RGB values and processes its features to compare the image with the training set. The output is the type of skin cancer the network deems the lesion to be.  

In the CNN with transfer learning, the network uses a pre-trained model known as VGG16, however, since the input minimum for VGG16 is 48x48, the input image of the lesion is resized to compensate for said requirements. 

## System/ Algorithm Design
The system is a composition of data preprocessing, splitting the data into train and test sets, then running the model on the data. Two models are employed: CNN with and without transfer learning. Both are quite similar in their approach, yet with important differences. 

### CNN / No Transfer Learning
•	Trains network from predetermined training images acquired in this case from Kaggle.com
•	Outputs a result after training on approximately 9000 images 
### CNN / Transfer Learning
•	Uses a pretrained model, in this case VGG16
•	Much longer processing time due to size of pretrained model 


## Experimental Evaluation 

### CNN / No Transfer Learning
•	Trained on Dataset from Kaggle.com ( skin lesion set with approximately 9000 test images )
•	Data cleaned (removed NA, dropped extra columns)
•	All images resized to 32x32 for improved speed
•	Features encoded
•	Data split into train and test set ( 75 / 25 )
•	Combinations of relu/sigmoid/tanh/Adam/SGD tried
•	Average time to process epoch ( 7s ) 

### CNN / Transfer Learning
•	Trained on VGG16 imageset ( approximately 1 million images and 16 layers )
•	Data cleaned (removed NA, dropped extra columns)
•	All images resized to 64x64 to fit into VGG16 
•	Features encoded
•	Data split into train and test set ( 75 / 25 )
•	Combinations of relu/sigmoid/tanh/Adam/SGD tried
•	Average time to process epoch ( 120s )

## Results

### CNN / No Transfer Learning

![image](https://user-images.githubusercontent.com/13923942/99602897-2287a100-29b7-11eb-81d8-bf0f23b47d3c.png)


#### The Relu / Adam combination produced the most accurate results. 

### Here we test image samples vs their true labels: 

![image](https://user-images.githubusercontent.com/13923942/99603067-6da1b400-29b7-11eb-8819-7af5e9e71f65.png)

![image](https://user-images.githubusercontent.com/13923942/99603091-77c3b280-29b7-11eb-8067-d07763ac4d23.png)

![image](https://user-images.githubusercontent.com/13923942/99603110-7eeac080-29b7-11eb-8c1f-3723171bdf90.png)

![image](https://user-images.githubusercontent.com/13923942/99603117-84480b00-29b7-11eb-870e-efc0a430b7c1.png)

![image](https://user-images.githubusercontent.com/13923942/99603126-89a55580-29b7-11eb-9f87-08f0770eb454.png)







