import numpy as np

from abc import ABC, abstractmethod
from collections import Counter


class Chainer(ABC):
    """ chainer class for chaining text transformations """

    @abstractmethod
    def process(self, data, chronicle):
        """ chain method for data pre-processing """
        
        pass

    def apply(self, data, chronicle):
        """ default action """

        return self.process(data, chronicle)


class Chronicle:

    def __init__(self, sequence_max = None, word_to_int = None, int_to_word = None, vocabulary_size = None, mean = None, std = None):

        self.sequence_max = sequence_max
        self.word_to_int = word_to_int
        self.int_to_word = int_to_word
        self.vocabulary_size = vocabulary_size
        self.mean = mean
        self.std = std

    def copy(self):

        # create an independent chronicle
        return Chronicle(sequence_max = self.sequence_max, word_to_int = self.word_to_int,
                         int_to_word = self.int_to_word, vocabulary_size = self.vocabulary_size, mean = self.mean,
                         std = self.std)


class Tokenize(Chainer):

    def process(self, data, chronicle):
        """ chain method for data pre-processing """

        # tokenize, split a sentence by space
        data = data.str.split()

        # find maximum size of sequence of tokens
        chronicle.sequence_max = max([ len(sequence) for sequence in data ])

        return data, chronicle


class Vocabulary(Chainer):

    def process(self, data, chronicle):
        """ create the known vocabulary basis """

        # count token occurrences
        chronicle.tokens = Counter([ token for sequence in data for token in sequence ])

        # vocabulary_size
        chronicle.vocabulary_size = len(chronicle.tokens) + 1

        # word to integer mapping, 0 is reserved for padding
        chronicle.word_to_int = { key : (index + 1) for index, key in enumerate(chronicle.tokens) }

        # integer to word mapping
        chronicle.int_to_word = { index : word for word, index in chronicle.word_to_int.items() }

        return data, chronicle

    def apply(self, data, chronicle):
        """ override apply transformation method """

        # nothing to do
        return data, chronicle


class NumericToToken(Chainer):

    def process(self, data, chronicle):
        """ apply textual transformation """

        assert(hasattr(chronicle, 'int_to_word'))

        # transform from textual to numerical representation
        data = [
            [ chronicle.int_to_word[token] for token in sequence if token in chronicle.int_to_word ]
        for sequence in data ]

        return data, chronicle


class TokenToNumeric(Chainer):

    def process(self, data, chronicle):
        """ apply numerical transformation """

        assert(hasattr(chronicle, 'word_to_int'))

        # transform from textual to numerical representation
        data = [
            [ chronicle.word_to_int[token] if token in chronicle.word_to_int else 0 for token in sequence ]
        for sequence in data ]

        return data, chronicle


class Filler(Chainer):

    def process(self, data, chronicle):
        """ apply padding to numerical content """

        # assert numerical representation of input data
        assert(all(isinstance(token, int) for sequence in data for token in sequence))
        assert(hasattr(chronicle, 'sequence_max'))

        # get the real size of each sequence
        chronicle.sizes = np.array([ len(sequence) - 1 for sequence in data ], dtype = "longlong")

        # transform by padding
        data = np.array([ seq + [0] * (chronicle.sequence_max - len(seq)) for seq in data ], dtype = "longlong")

        return data, chronicle


class Normalization(Chainer):

    def process(self, data, chronicle):

        # compute dispersion
        chronicle.std = np.std(data)

        # compute mean
        chronicle.mean = np.mean(data)

        # normalize data
        data = np.array((data - chronicle.mean) / chronicle.std, dtype = "float32")

        return data, chronicle

    def apply(self, data, chronicle):

        assert(chronicle.std is not None and chronicle.mean is not None)

        return self.process(data, chronicle)

    @staticmethod
    def scale(self, data, chronicle):

        # undo the normalization
        data = np.array(data * chronicle.std + chronicle.mean, dtype = "float32")

        return data, chronicle


class Composer(Chainer):

    def __init__(self, transforms):

        # current transformations
        self.transforms = transforms

        # check each transformation
        for transform in self.transforms:

            # check if it's chainer transformer
            if not isinstance(transform, Chainer):

                # list item is not an instance of Chainer transformer
                raise Exception("Illegal parameter, provide contiguous set of Chainer(s)")

        # initialize chronicle of transformations
        self.chronicle = Chronicle()

    def process(self, raw, chronicle):

        # initialize chainer data
        data = raw.copy()

        # apply transformations in series
        for transform in self.transforms:

            # process data
            data, _ = transform.process(data, self.chronicle)

        return data, self.chronicle

    def apply(self, raw, chronicle):

        # create an independent chronicle
        chronicle = self.chronicle.copy()

        # initialize chainer data
        data = raw.copy()

        # apply transformations in series
        for transform in self.transforms:

            # apply transform to data
            data, _ = transform.apply(data, chronicle)

        return data, chronicle
