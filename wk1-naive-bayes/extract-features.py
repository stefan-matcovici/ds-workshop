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


def generate_features(from_directory, bag, to_file, option, start):
    tokens = list(dict(bag).keys())
    with open(to_file, option) as output:
        final = 0
        for doc_number, file in enumerate(sorted(os.listdir(from_directory)), start):
            final = doc_number
            file_path = os.path.join(from_directory, file)
            with open(file_path) as f:
                words = list(filter(lambda x: len(x) > 3, f.read().split(" ")))
                words_counter = dict(Counter(words).most_common())
                for word in list(words_counter.keys()):
                    if word in tokens:
                        output.write(str(doc_number) + " " + str(tokens.index(word) + 1) + " " + str(words_counter[word]) + "\n")
    return final


if __name__ == "__main__":
    bag_of_words = Counter()

    words_t = 0

    add_to_bag_of_words(NONSPAM_TRAIN_DIRECTORY, bag_of_words)
    add_to_bag_of_words(SPAM_TRAIN_DIRECTORY, bag_of_words)
    add_to_bag_of_words(NONSPAM_TEST_DIRECTORY, bag_of_words)
    add_to_bag_of_words(SPAM_TEST_DIRECTORY, bag_of_words)

    bag = bag_of_words.most_common(2500)

    docs = generate_features(NONSPAM_TRAIN_DIRECTORY, bag, "train_features.txt", "w+", 1)
    docs = generate_features(SPAM_TRAIN_DIRECTORY, bag, "train_features.txt", "a", docs+1)

    docs = generate_features(NONSPAM_TEST_DIRECTORY, bag, "test_features.txt", "w+", 1)
    docs = generate_features(SPAM_TEST_DIRECTORY, bag, "test_features.txt", "a", docs+1)
