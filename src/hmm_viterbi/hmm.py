import numpy as np
import itertools as it


class HiddenMarkovModel:

    def __init__(self, sentences, algorithm='viterbi'):
        """
        :param sentences: array of (word, pos) pairs
        :param algorithm: 'exhaustive' or 'viterbi'
        """

        self.algorithm = algorithm

        list_of_all_words = [pair for line in sentences for pair in line]
        self.unique_words = sorted(list({pair[0] for pair in list_of_all_words}))
        self.unique_pos = sorted(list({pair[1] for pair in list_of_all_words}))

        self.pos_to_idx = {pos: idx for idx, pos in enumerate(self.unique_pos)}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.unique_words)}
        self.idx_to_pos = {v: k for k, v in self.pos_to_idx.items()}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        self.n_observations = len(self.unique_words)
        self.n_hidden_states = len(self.unique_pos)

        self.initial_probabilities = np.zeros(self.n_hidden_states)
        for line in sentences:
            pair = line[0]
            self.initial_probabilities[self.pos_to_idx[pair[1]]] += 1
        self.initial_probabilities /= len(sentences)

        self.transition_probabilities = np.zeros((self.n_hidden_states, self.n_hidden_states))
        for line in sentences:
            for first, second in zip(line[:-1], line[1:]):
                self.transition_probabilities[self.pos_to_idx[first[1]], self.pos_to_idx[second[1]]] += 1
        self.transition_probabilities /= np.sum(self.transition_probabilities, axis=1, keepdims=True)
        self.transition_probabilities[np.isnan(self.transition_probabilities)] = 1 / self.transition_probabilities.shape[1]

        self.emission_probabilities = np.zeros((self.n_hidden_states, self.n_observations))
        for pair in list_of_all_words:
            self.emission_probabilities[self.pos_to_idx[pair[1]], self.word_to_idx[pair[0]]] += 1
        self.emission_probabilities /= np.sum(self.emission_probabilities, axis=1, keepdims=True)
        self.emission_probabilities[np.isnan(self.emission_probabilities)] = 1 / self.emission_probabilities.shape[1]

    def predict(self, sentence):
        if self.algorithm == "exhaustive":
            return self._exhaustive(sentence)
        elif self.algorithm == "viterbi":
            return self._viterbi(sentence)
        raise ValueError("Unknown algorithm: " + self.algorithm)

    def _exhaustive(self, sentence):

        pos_ids = self.idx_to_pos.keys()

        best_combination = None
        best_probability = 0.0

        for combination in it.product(pos_ids, repeat=len(sentence)):
            # Here, `combination` is an array of POS tags; one for each word in the sentence
            # the product function generates all possible combinations of such tags

            # We start with 1.0, so that we can multiply probabilities later on
            probability = 1.0

            # Apply initial probability
            probability *= self.initial_probabilities[combination[0]]

            # Apply transition probabilities
            for first, second in zip(combination[:-1], combination[1:]):
                probability *= self.transition_probabilities[first, second]

            # Apply emission probabilities
            for word, pos in zip(sentence, combination):
                probability *= self.emission_probabilities[pos, self.word_to_idx[word]]

            if probability > best_probability:
                best_probability = probability
                best_combination = combination

        return tuple(self.idx_to_pos[id] for id in best_combination)

    def _viterbi(self, sentence):
        ...
