import numpy as np

TRAIN_FEATURES_FILE = "./prepared-data/train-features.txt"
TRAIN_LABELS_FILE = "./prepared-data/train-labels.txt"
TEST_FEATURES_FILE = "./prepared-data/test-features.txt"
TEST_LABELS_FILE = "./prepared-data/test-labels.txt"

numTrainDocs = 700
numTestDocs = 260
numTokens = 2500
nuTestTokens = 2500


def load_features(file_name, no_docs, no_tokens):
    x = np.zeros((no_docs, no_tokens))

    with open(file_name) as f:
        for line in f.read().split("\n"):
            document, word_id, appeareances = list(map(int, line.split(" ")))
            x[document - 1, word_id - 1] = appeareances

    return x


def load_labels(file_name):
    with open(file_name) as f:
        labels = np.array(list(map(int, f.read().split("\n"))))
        labels = np.array([True if x == 1 else False for x in labels])
    return labels


if __name__ == "__main__":
    x = load_features(TRAIN_FEATURES_FILE, numTrainDocs, numTokens)
    train_labels = load_labels(TRAIN_LABELS_FILE)

    spam_documents = [i for i, x in enumerate(train_labels) if x == 1]
    non_spam_documents = [i for i, x in enumerate(train_labels) if x == 0]

    spam_probability = len(spam_documents) / numTrainDocs
    email_lenghts = np.sum(x, 1)

    spam_wc = np.sum(np.take(email_lenghts, spam_documents))
    nonspam_wc = np.sum(np.take(email_lenghts, non_spam_documents))

    prob_tokens_spam = (np.sum(np.take(x, spam_documents, 0), 0) + 1) / (spam_wc + numTokens)
    prob_tokens_nonspam = (np.sum(np.take(x, non_spam_documents, 0), 0) + 1) / (nonspam_wc + numTokens)

    prob_tokens_spam = prob_tokens_spam.reshape(-1, 1)
    prob_tokens_nonspam = prob_tokens_nonspam.reshape((-1, 1))

    test_x = load_features(TEST_FEATURES_FILE, numTestDocs, nuTestTokens)

    log_a = np.dot(test_x, np.log(prob_tokens_spam)) + np.log(spam_probability)
    log_b = np.dot(test_x, np.log(prob_tokens_nonspam)) + np.log(1 - spam_probability)
    output = log_a > log_b

    test_labels = load_labels(TEST_LABELS_FILE)
    test_labels = test_labels.reshape((-1, 1))

    numdocs_wrong = np.sum(np.logical_xor(output, test_labels))
    fraction_wrong = numdocs_wrong / numTestDocs

    print(numdocs_wrong)
