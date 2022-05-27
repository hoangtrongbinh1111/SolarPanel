***Guide to run code***

1. Split dataset to train, val, test with ratio 7:2:1 
- Command: python divideDataset.py
2. [Optional] Statistical your dataset
- Command: python visualizeData.py
3. [Optional] Augmention data to solve imbalance data
- Command: python imbalanceData.py
4. Run transfer learning to classify your image into labels. We use VGG19, InceptionV3 and MobileNet
- Command: python transferLearning.py
5. [Main code] Run custom CNN based on backbone Resnet
- Command: python customCNN.py
6. [Main code] Run SpinalNet
- Command: python spinalNet.py
7. [Ensemble Learning] Using CNN as feature extractor and some traditional methods such as KNN, SVM, 
Decision Tree, Random Forest to evaluate
- Command: python patternRecognition.py

***Note***
- Your checkpoints will be saved in folder ./checkpoint
- Your logs will be saved in folder ./logs
- Your training history will be saved in folder ./history
- Your dataset will be saved in folder ./dataset/dataset12ClassKeras

***Author***
- Hoang Trong Binh - CNTT1K53
