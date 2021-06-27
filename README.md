# Skin Cancer Detection
Can we detect skin cancer through artificial intelligence?


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

![image](https://user-images.githubusercontent.com/13923942/99603067-6da1b400-29b7-11eb-8819-7af5e9e71f65.png) : Correct

![image](https://user-images.githubusercontent.com/13923942/99603091-77c3b280-29b7-11eb-8067-d07763ac4d23.png) : Incorrect 

![image](https://user-images.githubusercontent.com/13923942/99603110-7eeac080-29b7-11eb-8c1f-3723171bdf90.png) : Correct 

![image](https://user-images.githubusercontent.com/13923942/99603117-84480b00-29b7-11eb-870e-efc0a430b7c1.png) : Correct

![image](https://user-images.githubusercontent.com/13923942/99603126-89a55580-29b7-11eb-9f87-08f0770eb454.png) : Correct
 


### CNN / Transfer Learning

![image](https://user-images.githubusercontent.com/13923942/99603205-b0fc2280-29b7-11eb-90da-2b1e377753cc.png)


#### The Relu / Adam combination produced the most accurate results. 

### Here we test image samples vs their true labels: 

![image](https://user-images.githubusercontent.com/13923942/99603247-c5401f80-29b7-11eb-98e0-4e0af152b715.png) : Correct

![image](https://user-images.githubusercontent.com/13923942/99603260-cbce9700-29b7-11eb-9a33-159221902bc2.png) : Incorrect 

![image](https://user-images.githubusercontent.com/13923942/99603307-e0ab2a80-29b7-11eb-8878-33f11cfc1fc8.png) : Correct 

![image](https://user-images.githubusercontent.com/13923942/99603328-e739a200-29b7-11eb-99d6-1dfe1f8d4480.png) : Correct

![image](https://user-images.githubusercontent.com/13923942/99603126-89a55580-29b7-11eb-9f87-08f0770eb454.png) : Correct



# Conclusion 
Based upon the results, the neural network was accurate approximately 70% of the time. The system was more accurate at  predicting Vascular Lesions, Melanoma, and Actinic Keratoses, simply due to the disproportionate dataset. 

Comparing the two models, the CNN without transfer learning had significantly faster runtime. The tests were run on my local machine, which might be the reason for why the CNN with transfer learning took almost 15x longer to complete training. However, even with the extra time to compile, the accuracy between the two models was quite similar, approximately, though the model with transfer learning made some predictions that were more accurate than the other model. All in all, in a practical sense, training the model with transfer learning ( more data ) should, in theory, improve the results, which is what can be seen here. 



# Learning Experience

At first, I was unsure of which dataset to use. I definitely am appreciative now of the immense datasets available online to use for our models. If it weren’t for that, finding suitable data to use would have been a nightmare. My first couple of days trying to find data were unsuccessful, so I attempted to webscrape a few sites for their images, and the issue arose in that the images were either all watermarked, not labeled, or a combination of both.

I also learned of the importance of using AI to garner accurate results. In a situation like this, which if a patient were to rely on to detect skin cancer, the results may or may not be suitable to use. Especially when death is a potential outcome for an incorrect result, relying on AI to detect something like skin cancer can be risky if the model if not the most pristine, suited model. Therefore, I have definitely come to see the vital nature of finding accurate data and applying it correctly.   


