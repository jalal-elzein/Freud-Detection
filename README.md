# Introduction 
This is the repository of my project for the CSC461 - Machine Learning course I undertook. I completed this project alongside [Tamer Kobba]() and [Hussein Fadel](). Read the report for more information about the project and journey.

# Environment 
This code was tested in Python 3.10

The required packages can be found in the `requirements.txt` file and can be installed using the following command:
```shell
pip install -r requirements.txt
```

# Examples to run
```shell
python main.py --save_flag 1 --model_filename demo_model.keras
```

Take a look at the args in `main.py` for more customization of your runs

# Results
We managed to achieve over 99% accuracy and 99% AUC
```
582/582 [==============================] - 1s 2ms/step
----- nn_sbert_93k.keras -----
Accuracy: 0.990
Precision: 0.990
Recall: 0.989
F1 Score: 0.989
Confusion Matrix:
[[ 7905   118]
 [   76 10522]]
```