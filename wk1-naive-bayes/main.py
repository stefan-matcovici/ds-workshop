import numpy as np

numTrainDocs = 700
numTestDocs = 260
numTokens = 2500
nuTestTokens = 2500

if __name__ == "__main__":
    x = np.zeros((numTrainDocs, numTokens))

    with open("./prepared-data/train-features.txt") as f:
        for line in f.read().split("\n"):
            document, word_id, appeareances = list(map(int, line.split(" ")))
            x[document-1, word_id-1] = appeareances

    spam_documents = []
    non_spam_documents = []
    with open("./prepared-data/train-labels.txt") as f:
        for line_no, label in enumerate(f.read().split("\n"), 0):
            if label == "0":
                non_spam_documents.append(line_no)
            else:
                spam_documents.append(line_no)

    spam_probability = len(spam_documents) / numTrainDocs
    email_lenghts = np.sum(x, 1)

    spam_wc = np.sum(np.take(email_lenghts, spam_documents))
    nonspam_wc = np.sum(np.take(email_lenghts, non_spam_documents))

    prob_tokens_spam = (np.sum(np.take(x, spam_documents, 0), 0) + 1) / (spam_wc + numTokens)
    prob_tokens_nonspam = (np.sum(np.take(x, non_spam_documents, 0), 0) + 1) / (nonspam_wc + numTokens)

    prob_tokens_spam = prob_tokens_spam.reshape(-1, 1)
    prob_tokens_nonspam = prob_tokens_nonspam.reshape((-1, 1))

    test_x = np.zeros((numTestDocs, nuTestTokens))
    with open("./prepared-data/test-features.txt") as f:
        for line in f.read().split("\n"):
            document, word_id, appeareances = list(map(int, line.split(" ")))
            test_x[document-1, word_id-1] = appeareances

    log_a = np.dot(test_x, np.log(prob_tokens_spam)) + np.log(spam_probability)
    log_b = np.dot(test_x, np.log(prob_tokens_nonspam)) + np.log(1 - spam_probability)
    output = log_a > log_b

    with open("./prepared-data/test-labels.txt") as f:
        test_labels = np.array(list(map(int, f.read().split("\n"))))
        test_labels = np.array([True if x == 1 else False for x in test_labels])

    test_labels = test_labels.reshape((len(test_labels), 1))

    test_labels = np.reshape(test_labels, (len(test_labels), 1))
    numdocs_wrong = np.sum(np.logical_xor(output, test_labels))
    fraction_wrong = numdocs_wrong / numTestDocs

    print(numdocs_wrong)
