from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

TRAIN_FEATURES_FILE = "./data/train.data.txt"
TRAIN_LABELS_FILE = "./data/train.label.txt"
TEST_FEATURES_FILE = "./data/test.data.txt"
TEST_LABELS_FILE = "./data/test.label.txt"

numTrainDocs = 11269
numTestDocs = 7505
numTokens = 61188
numTestTokens = 61188


def load_features(file_name, no_docs, no_tokens):
    x = np.zeros((no_docs, no_tokens), dtype="float64")

    with open(file_name) as f:
        for line in f.read().split("\n")[:-1]:
            document, word_id, appeareances = list(map(int, line.split(" ")))
            x[document - 1, word_id - 1] = appeareances

    return x


def load_labels(file_name):
    with open(file_name) as f:
        labels = np.array(list(map(int, f.read().split("\n")[:-1])))
    return labels


def load_class_names(file_name):
    with open(file_name) as f:
        classes = np.array(f.read().split("\n")[:-1])
    return classes


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout()


dirichlet_factor = 1e-2


def plot_confusion_matrix_for_output(classes, test_labels, output):
    cnf_matrix = confusion_matrix(test_labels, output)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


if __name__ == "__main__":
    x = load_features(TRAIN_FEATURES_FILE, numTrainDocs, numTokens)
    train_labels = load_labels(TRAIN_LABELS_FILE)

    test_x = load_features(TEST_FEATURES_FILE, numTestDocs, numTestTokens)
    test_labels = load_labels(TEST_LABELS_FILE)

    noTypeDocs = len(set(train_labels))

    docs_prob_distribution = [v / numTrainDocs for k, v in Counter(train_labels).items()]
    docs_idx = [np.where(train_labels == docType)[0] for docType in range(1, noTypeDocs + 1)]
    doc_lenghts = np.sum(x, 1)

    docs_word_count = [np.sum(np.take(doc_lenghts, docs_idx[docType])) for docType in range(noTypeDocs)]
    factor = 1
    plt.figure()
    factors = []
    accuracies = []
    while factor > 10 ** (-5):
        probs = [
            (np.sum(np.take(x, docs_idx[docType], 0), 0) + 1 * factor) / (docs_word_count[docType] + numTokens * factor)
            for docType in
            range(noTypeDocs)]

        logs = [np.dot(test_x, np.log(probs[docType])) + np.log(docs_prob_distribution[docType]) for docType in
                range(noTypeDocs)]
        output = np.argmax(logs, 0) + 1

        numdocs_right = np.sum([1 if o == t else 0 for o, t in zip(output, test_labels)])
        fraction_right = numdocs_right / numTestDocs

        print(fraction_right)
        factors.append(factor)
        accuracies.append(fraction_right)
        factor *= 0.5

    plt.plot(list(reversed(factors)), list(reversed(accuracies)), label="Accuracy")
    plt.legend()
    plt.show()

    classes = load_class_names("./data/newsgrouplabels.txt")

    # plot_confusion_matrix_for_output(classes, test_labels, output)
