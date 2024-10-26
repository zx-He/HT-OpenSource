# HT-OpenSource
the dataset and source code of the paper "HT-Auth: Unobtrusive VR Headset Authentication via Subtle Head Tremors“



Main Dataset：

​	Download from: https://www.kaggle.com/datasets/miracle0723/ht-dataset	

​	Data from 30 subjects(subject ID = folder name), each subject contributes 50 samples. 

​	Each sample has a length of 546, assembled by MFCC, Spectral Centroid, Spectral Spread,  Spectral Flux, Spectral Entropy, Zero-Cross Rate features.



Source code: 

​	Run "Siamese_transfer.py" to get the system performance of different subjects, the samples number utilized to train user-specific model is set to 10.

​    More details can be found in the source code.

​	

