# Kaggle TGS Salt Identification Challenge 34th Solution (34/3291)

# 1. Overall Solution (Worked for me)
## 1.1 Encoder
I have tried a lot of backbone, such as resnet34, senet, se-resnet, etc. But only Densenet and DPN worked for me.
I used three encoders in my final models:
* densenet without dilation
* densenet with dilation
* normal DPN

(More details can be found in the main_code_torch/models)
## 1.2 Decoder
* Unet style structure with hypercolumn (Heng's solution)
* linknet style sturcture without hypercolumn
* ModifiedSCSEBlock (the original scse block did not work for me)
## 1.3 Training
* adam (SGD did not work for me )
* (0.5) decay learning rate if the validation miou did not improve for 15 epoches
* Training 40 more epoches after got the highest miou on validation dataset.
* 10 folds for each model
* 5 cycle learning for each fold
* Heng's data sugmentation
* deep supervision
* lovasz loss
## 1.4 Pseudo label
Afraid of overfitting, I emsembled the results from densenet to train DPN. This improved 0.001 in public LB.
## 1.5 Ensemble
Densenet(with dilation)+Densenet(without dilation,two scale 128and224)+DPN+link_Densenet+Pseudo

Just average all of them.
## 1.6 Testing
Just flip horizontal TTA. I have tried some other TTA, did not work for me.

## 1.7 Post-processing
Modified some masks if they adjoin with vertical masks in the jigsaw.(just copy them)

(SPN, CRF, filtering small image blocks did not work for me)

# 2. Code
* main_code_torch/ml_train.py : train all folds
* main_code_torch/ml_eval.py  : offline validation
* main_code_torch/ml_test.py  : online testing and make submission
* main_code_torch/models      : all models I used in my final submission
