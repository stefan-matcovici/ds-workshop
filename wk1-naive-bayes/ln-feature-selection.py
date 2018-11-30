import math
import os
from collections import Counter

DATA_EMAILS_DIRECTORY = "data-emails"

NONSPAM_TRAIN_DIRECTORY = os.path.join(DATA_EMAILS_DIRECTORY, "nonspam-train")
SPAM_TRAIN_DIRECTORY = os.path.join(DATA_EMAILS_DIRECTORY, "spam-train")

NONSPAM_TEST_DIRECTORY = os.path.join(DATA_EMAILS_DIRECTORY, "nonspam-test")
SPAM_TEST_DIRECTORY = os.path.join(DATA_EMAILS_DIRECTORY, "spam-test")


def add_to_bag_of_words(directory, bag):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        with open(file_path) as f:
            words = filter(lambda x: len(x) > 3, f.read().split(" "))
            bag += Counter(words)


def generate_features(from_directory, m, to_file, option, start):
    keys = list(m.keys())
    with open(to_file, option) as output:
        final = 0
        for doc_number, file in enumerate(sorted(os.listdir(from_directory)), start):
            final = doc_number
            file_path = os.path.join(from_directory, file)
            with open(file_path) as f:
                words = list(filter(lambda x: len(x) > 3, f.read().split(" ")))
                words_counter = dict(Counter(words).most_common())
                for word in list(words_counter.keys()):
                    if word in m:
                        output.write(
                            str(doc_number) + " " + str(keys.index(word) + 1) + " " + str(words_counter[word]) + "\n")
    return final


THRESHOLD = [-1, 1]

if __name__ == "__main__":
    non_spam_bag_of_words = Counter()
    spam_bag_of_words = Counter()

    words_t = 0

    add_to_bag_of_words(NONSPAM_TRAIN_DIRECTORY, non_spam_bag_of_words)
    add_to_bag_of_words(NONSPAM_TEST_DIRECTORY, non_spam_bag_of_words)

    add_to_bag_of_words(SPAM_TRAIN_DIRECTORY, spam_bag_of_words)
    add_to_bag_of_words(SPAM_TEST_DIRECTORY, spam_bag_of_words)

    no_non_spam_words = len(non_spam_bag_of_words.keys())
    no_spam_words = len(spam_bag_of_words.keys())

    ln_map = {}
    for word in non_spam_bag_of_words.keys():
        app_non_spam = non_spam_bag_of_words[word]
        app_spam = spam_bag_of_words[word]

        if not (app_spam == 0 or app_non_spam == 0):
            value = math.log((app_spam / no_spam_words) / (app_non_spam / no_non_spam_words))
            if value < THRESHOLD[0] or value > THRESHOLD[1]:
                ln_map[word] = app_non_spam + app_spam

    print(len(ln_map))

    docs = generate_features(NONSPAM_TRAIN_DIRECTORY, ln_map, "ln_train_features.txt", "w+", 1)
    docs = generate_features(SPAM_TRAIN_DIRECTORY, ln_map, "ln_train_features.txt", "a", docs+1)

    docs = generate_features(NONSPAM_TEST_DIRECTORY, ln_map, "ln_test_features.txt", "w+", 1)
    docs = generate_features(SPAM_TEST_DIRECTORY, ln_map, "ln_test_features.txt", "a", docs+1)
