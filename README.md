Copyright (c) 2018/4/5 Zhiming Hu jimmyhu (at) pku.edu.cn All Rights Reserved.
The tensorflow CNN & Bilinear CNN codes for Oxford Flowers 17 Dataset.

Directories & Files:

'DataPreprocess' directory: stores the codes to preprocess the original images.
'Dataset' directory: stores the TFRecord files created in the DataPreprocess directory.
'Models' directory: stores our trained models.
'Results' directory: stores the learning curves of our model.
'OxFlowers_CNN_75.py': the main function of CNN model.
'OxFlowers_BCNN_85.py': the main function of Bilinear CNN model.

Environments:
Python 3.6+

tensorflow 1.4.1+

Usage:

Step 1: Check the 'Dataset/' directory to confirm whether the TFRecord files exist. 
If not, run the code in 'DataPreprocess/' to create the TFRecord files.

Step 2: Run 'OxFlowers_CNN_75.py' & 'OxFlowers_BCNN_85.py' to test the model.
The accuracy of 'OxFlowers_CNN_75.py' on the test set is 75.59%. 
The accuracy of 'OxFlowers_BCNN_85.py' on the test set is 85.00%. 
If the models do not exit, you can uncomment the training code in the main function to retrain the model.
