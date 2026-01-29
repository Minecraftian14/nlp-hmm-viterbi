import numpy as np


class HiddenMarkovModel:

    def __init__(self, sentences):
        # sentences is an array of (word, pos) pairs

        list_of_all_words = [pair for line in sentences for pair in line]
        unique_words = {pair[0] for pair in list_of_all_words}
        unique_pos = {pair[1] for pair in list_of_all_words}

        self.pos_to_idx = {pos: idx for idx, pos in enumerate(unique_pos)}
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_words)}

        self.n_observations = len(unique_words)
        self.n_hidden_states = len(unique_pos)

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

        self.emission_probabilities = np.zeros((self.n_hidden_states, self.n_observations))
        for pair in list_of_all_words:
            self.emission_probabilities[self.pos_to_idx[pair[1]], self.word_to_idx[pair[0]]] += 1
        self.emission_probabilities /= np.sum(self.emission_probabilities, axis=1, keepdims=True)
