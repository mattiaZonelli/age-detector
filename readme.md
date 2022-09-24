Age Prediction

* 3 modelli entrambi allenati su train (2236 imgaes) e validation (240 images):
  - 1 modello con 10 output neurons;
  - 1 modello con 20 output neurons;
  - 1 modello con 101 output neurons;
  
Dataset:
ChaLearn LAP, Apparent age V1 (ICCV '15)




**VAL. ACC.**\
TOP K   10classes   20classes   101classes\
1       0.3077		0.0769      0.0000\
2       0.6923      0.2308      0.0000\
3		0.7692		0.4615      0.0000

101classes revival\
@5:   0.0769\
@10:  0.3846\
@15:  0.6154

