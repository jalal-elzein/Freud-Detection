import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_curve, 
    auc
)
import seaborn as sns
import matplotlib.pyplot as plt


# ----- DATA -------
DATASET_FILENAME = "nn_dataset.csv"
EMBEDDINGS_FILENAME = "sbert_embeddings.npy" 
FRAC_OF_DATA = 0.1
FRAC_TEST = 0.2
SEED = 42
# ----- MODEL ------
BATCH_SIZE = 32
CLASSIFICATION_THRESHOLD = 0.5
EPOCHS = 10
MODEL_FILENAME = "nn_sbert_93k.keras"
MODEL_LOSS = "binary_crossentropy"
MODEL_METRICS = ["accuracy"]
MODEL_OPTIMIZER = "adam"
READ_OR_CONSTRUCT = 1



def draw_conf(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="magma", cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def draw_roc(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='purple', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



# reading raw data
raw_df = pd.read_csv(DATASET_FILENAME)
raw_embeddings = np.load(EMBEDDINGS_FILENAME)
raw_df["embedding"] = list(raw_embeddings)

# taking a sample
df = raw_df.sample(frac=FRAC_OF_DATA).reset_index(drop=True)

# extracting features and targets
embeddings = np.array(list(df["embedding"])) # should be list of np.array
labels = np.array(list(df["class"]))

# performing hold-out set data splitting
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=FRAC_TEST, random_state=SEED)


# constructing the model
if READ_OR_CONSTRUCT == 1:
    model = Sequential()
    model.add(Dense(128, input_dim=384, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=MODEL_LOSS, optimizer=MODEL_OPTIMIZER, metrics=MODEL_METRICS)

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.save(MODEL_FILENAME)
else:
    model = load_model(MODEL_FILENAME)

print(model.summary())


y_pred = model.predict(X_test)
y_pred_bin = (y_pred > CLASSIFICATION_THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, y_pred_bin)
precision = precision_score(y_test, y_pred_bin, average='macro')
recall = recall_score(y_test, y_pred_bin, average='macro')
f1 = f1_score(y_test, y_pred_bin, average='macro')
conf_matrix = confusion_matrix(y_test, y_pred_bin)

print('Accuracy: %.3f' % accuracy)
print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)
print('F1 Score: %.3f' % f1)
print('Confusion Matrix:')
draw_conf(conf_matrix)