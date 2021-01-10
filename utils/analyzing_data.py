import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def plot_confusion_matrix(cm, class_names, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.round(decimals=2)

    figure = plt.figure(figsize=(30, 25))
    df_cm = pd.DataFrame(cm)  # , index=class_names, columns=class_names
    sn.set(font_scale=4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 60}, cmap=plt.cm.Blues, fmt='g')  # font size
    plt.title(title)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, horizontalalignment='center', rotation=30)
    plt.yticks(tick_marks, class_names, verticalalignment='top', rotation=30)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def multiclass_analysis(model, test_ds, class_names, save_fig=None):
    test_list = []
    labels_list = []
    preds_list = []
    results_classes = []
    to_print = ""
    nclasses = len(class_names)

    # Convert test_ds to python list and split in labels and test set
    # NB. The labels are converted to the argmax()
    for s in test_ds:
        labels_list.append(np.argmax(s[1].numpy()[0].tolist()))
        test_list.append(s[0].numpy()[0].tolist())

    # Get predictions for data in test set and convert predictions to python list
    preds = model.predict(test_list)
    for p in preds:
        preds_list.append(np.argmax(p.tolist()))

    # Calculate Confusion Matrix
    cm = tf.math.confusion_matrix(labels_list, preds_list, num_classes=nclasses).numpy()
    to_print += np.array2string(cm) + "  \n"

    # Print the confusion matrixes (normalized and not)
    plot_confusion_matrix(confusion_matrix(labels_list, preds_list, normalize='true'), class_names=class_names,
                          title="Confusion Matrix Normalized")
    # plt.save() save the latest plot created
    if save_fig is not None:
        plt.savefig(save_fig + "_NORMALIZED.png")
    plot_confusion_matrix(confusion_matrix(labels_list, preds_list), class_names=class_names, title="Confusion Matrix")
    if save_fig is not None:
        plt.savefig(save_fig + ".png")

    # Compute ROC curve and ROC area for each class
    # Adopting One vs All approach: per each class x, the label x becomes 1 and all the other labels become 0
    for i in range(nclasses):
        i_label_list = []
        i_pred_list = []
        # Converting labels to 1 if i or 0 otherwise
        for k in range(len(labels_list)):
            i_label_list.append(1 if labels_list[k] == i else 0)
            i_pred_list.append(1 if preds_list[k] == i else 0)

        # Calculating roc_curve and auc starting from false positive rate and true positive rate
        fpr, tpr, thresholds = roc_curve(i_label_list, i_pred_list)
        roc_auc = auc(fpr, tpr)
        results_classes.append({'AUC': roc_auc, 'ROC': [fpr, tpr, thresholds]})

    '''
    Examples of CM for multi-label classification | and index
    NB. Taking into account class at (12) as True Positive
    TN TN FP TN | 00 01 02 03
    FN FN TP FN | 10 11 12 12
    TN TN FP TN | 20 21 22 23
    TN TN FP TN | 30 31 32 33
    '''
    for ind in range(nclasses):
        TP = TN = FP = FN = 0
        for i in range(nclasses):
            for j in range(nclasses):
                if i == j and i == ind:
                    TP = cm[i][j]
                elif ind == j:
                    FP += cm[i][j]
                elif ind == i:
                    FN += cm[i][j]
                else:
                    TN += cm[i][j]
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = (2*precision*recall)/(precision+recall)
        results_classes[ind].update({'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
                                    'acc': accuracy, 'prec': precision, 'rec': recall, 'fm': f1})
        to_print += "class {} -> TP: {}, TN: {}, FP: {}, FN: {}\n\tacc: {}, prec: {}, rec: {}, fm: {}, auc: {}\n"\
            .format(class_names[ind], TP, TN, FP, FN, accuracy, precision, recall, f1, results_classes[ind]['AUC'])

    return cm, results_classes, to_print
