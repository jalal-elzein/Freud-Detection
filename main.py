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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_filename', type=str, default="nn_dataset.csv", help='path or name of file where dataset csv is stored')
parser.add_argument('--embeddings_filename', type=str, default="sbert_embeddings.npy", help="path of name of file where embeddings numpy binary is stored")
parser.add_argument("--frac_of_data", type=float, default=0.1, help="percentage of data to be used")
parser.add_argument("--frac_test", type=float, default=0.2, help="percentage of data to be used as a test split")
parser.add_argument("--seed", type=int, default=42, help="seed for randomness")
parser.add_argument("--batch_size", type=int, default=32, help="size of data batches to be processed")
parser.add_argument("--classification_threshold", type=float, default=0.5, help="what percentage is the cutoff between a positive and a negative prediction")
parser.add_argument("--epochs", type=int, default=10, help="how many epochs the model will train for")
parser.add_argument("--save_flag", type=int, default=0, help="whether or not to save the model on disk")
parser.add_argument("--read_flag", type=int, default=0, help="whether or not to read the model from disk")
parser.add_argument("--model_filename", type=str, help="file name or path where the model is or should be saved")
parser.add_argument("--model_loss", type=str, default="binary_crossentropy", help="loss function the model is optimizing")
parser.add_argument("--model_metrics", type=str, default='["accuracy"]', help="metrics for the model")
parser.add_argument("--model_optimizer", type=str, default="adam", help="optimizer for the model")
args = parser.parse_args()


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
raw_df = pd.read_csv(args.dataset_filename)
raw_embeddings = np.load(args.embeddings_filename)
raw_df["embedding"] = list(raw_embeddings)

# taking a sample
df = raw_df.sample(frac=args.frac_of_data).reset_index(drop=True)

# extracting features and targets
embeddings = np.array(list(df["embedding"])) # should be list of np.array
labels = np.array(list(df["class"]))

# performing hold-out set data splitting
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=args.frac_test, random_state=args.seed)


# constructing the model
if args.read_flag == 0:
    model = Sequential()
    model.add(Dense(128, input_dim=384, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=args.model_loss, optimizer=args.model_optimizer, metrics=eval(args.model_metrics))

    model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    if args.save_flag == 1:
        model.save(args.model_filename)
else:
    model = load_model(args.model_filename)

print(model.summary())


y_pred = model.predict(X_test)
y_pred_bin = (y_pred > args.classification_threshold).astype(int)

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