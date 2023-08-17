Age Prediction

* 3 models all trained with training set (2236 images) and validation set (240 images):
  - 1 model with 10 output neurons;
  - 1 model with 20 output neurons;
  - 1 model with 101 output neurons;
  
Dataset:
ChaLearn LAP, Apparent age V1 (ICCV '15)

It uses a pre-trained VGG16 and does transfer learning and fine-tuning on age prediction. In this way, the training process should be faster and performances should be boosted.\
The last fully connected layer has been changed to 10,20 or 101 neurons depending on how many classes we want.


**VAL. ACC.**\
TOP K   10classes   20classes   101classes\
1       0.3077		  0.0769      0.0000\
2       0.6923      0.2308      0.0000\
3		    0.7692		  0.4615      0.0000

In the tl_fl.ipynb is a version made with tensorflow, but we does transfer learning and fine-tuning on Resnet50.
