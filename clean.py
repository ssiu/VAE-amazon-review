"""
script for text cleaning
"""
import re
import utils
import nltk
from nltk.stem.wordnet import WordNetLemmatizer


def clean_corpus(file_path):
    sents = []
    data = utils.load_dataset(file_path)
    for l in data:
        x = l.lower()
        x = remove_abbr(x)
        x = remove_numbers(x)
        x = remove_misc(x)
        token = nltk.word_tokenize(x)
        token = [remove_plural(word) for word in token]
        sents.append(token + ['</s>'])
    return sents


def clean_corpus1(file_path):
    """
    Reads a JSON file and clean the text data

    Args
        file_path: JSON file

    Returns
        data (List[List[str]]): List containing sentences
    """
    sents = []
    data = []
    # './datasets/reviews/Magazine_Subscriptions_5.json.gz'
    with gzip.open(file_path) as f:
        for l in f:
            x = json.loads(l.strip())
            if 'reviewText' not in x:
                continue
            sent = x['reviewText'].lower()
            sent = remove_abbr(sent)
            sent = remove_numbers(sent)
            sent = remove_misc(sent)
            sents.append(sent)
    for sent in sents:
        token = nltk.word_tokenize(sent)
        token = [remove_plural(word) for word in token]
        data.append(token + ['</s>'])
    return data


def remove_plural(word):
    lem = WordNetLemmatizer()
    return lem.lemmatize(word)


def remove_abbr(sent):
    # need to remove 'car/truck' to 'car or truck'
    sent = re.sub(r"'ll", " will", sent)
    sent = re.sub(r"'ve", " have", sent)
    sent = re.sub(r"'m", " am", sent)
    sent = re.sub(r"'d", " would", sent)
    sent = re.sub(r"'re", " are", sent)
    sent = re.sub(r"can't", "can not", sent)
    sent = re.sub(r"won't", "will not", sent)
    sent = re.sub(r"n't", " not", sent)
    sent = re.sub("#", " number ", sent)
    sent = re.sub(" has ", " have ", sent)
    sent = re.sub(" was ", " is ", sent)
    sent = re.sub(" does ", " do ", sent)
    return sent


def remove_numbers(sent):
    """
    Remove
    dates 5/6/2001
    fractions 1/3
    big numbers 1,000,000

    Convert them all to '#num'
    """
    sent = re.sub("\d+|\d+,\d+|\d+,\d+,\d+|\d+,\d+,\d+,\d+|\d+\.\d*|\d+/\d+", " #num ", sent)
    return sent


def remove_misc(sent):
    #separate . from next word
    sent = re.sub("\.\.\.|\.\.|\.", " . ", sent)
    # remove * around words
    sent = re.sub("\*", "", sent)
    # remove dates
    #remove '
    sent = re.sub("[()<>\-\"\']", " ", sent)
    sent = re.sub("/", " and ", sent)
    return sent

