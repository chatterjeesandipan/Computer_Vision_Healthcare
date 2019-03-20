# Chest X-ray Classification Problem (Kaggle)
Using VGG and ResNet style computer vision architectures for simple object classification problems

I am new to the field of computer vision and ML/DL. Hence, I mostly refer to the Youtube videos on the above topics and practice my concepts on Kaggle datasets. I am a beginner in this field and any help from you (for this project or in general, Python and ML/DL) would be appreciated.

I recently came across the chest X-ray Pneumonia classification dataset on Kaggle. You can download the dataset from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. The dataset contains the training, validation and testing images. You can treat this as a 2-class problem (normal or pneumonia) or a 3-class problem (normal or bacterial pneumonia or viral pneumonia).

In my code, I have considered both the problems (2 or 3 classes). In particular, the 2-class problem shows a highly unbalanced dataset where the "normal" category appears almost 3 times less than the "Pneumonia" category. The unbalanced data was dealt with using a Keras ImageDataGenerator. Further, the vision/feature extraction model is based on the standard VGG19 sequential model. The model summary shows a total of about 12 million parameters. The callback features were also implemented, which allowed me to save the best model based on the validation accuracy. The accuracy of the model is about 80%, along with a validation accuracy of about 75%, after 30 epochs. These numbers may seem low, but I am still learning the proper technqiues of hyper-parameter tuning to improve these numbers. If you see any scope of improvement, please let me know. Thanks!

