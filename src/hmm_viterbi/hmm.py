import numpy as np
import itertools as it
import collections as cls

from .scoring import micro_accuracy_score


def smoothen(matrix, smoothing):
    return matrix * (1 - smoothing) + smoothing / matrix.shape[-1]


class HiddenMarkovModel:

    def __init__(self, sentences, algorithm='viterbi', smoothing=0.01, rarity_factor=0.05):
        """
        :param sentences: list of (word, pos) pairs
        :param algorithm: 'exhaustive' or 'viterbi'
        """

        self.algorithm = algorithm
        self.smoothing = smoothing

        list_of_all_pairs = [pair for line in sentences for pair in line]
        counter = cls.Counter([word for word, _ in list_of_all_pairs])
        threshold_rank = max(
            # Any values that occur less than 5% in the corpus
            len([x for x in counter.values() if np.log(x) < np.log(counter.most_common()[0][1]) * rarity_factor]),
            # At least pick up some words
            int(len(counter) ** 0.5)
        )
        knowledge = {word for word, freq in counter.most_common()[:(1 + len(counter) - threshold_rank)]}
        list_of_all_pairs = [
            ((word, pos) if word in knowledge else ("UNKNOWN", pos))
            for word, pos in list_of_all_pairs]
        sentences = [
            [
                ((word, pos) if word in knowledge else ("UNKNOWN", pos))
                for (word, pos) in sentence]
            for sentence in sentences]

        self.unique_words = sorted(list({word for word, pos in list_of_all_pairs}))
        self.unique_pos = sorted(list({pos for word, pos in list_of_all_pairs}))

        self.pos_to_idx = {pos: idx for idx, pos in enumerate(self.unique_pos)}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.unique_words)}
        self.idx_to_pos = {v: k for k, v in self.pos_to_idx.items()}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        self.n_observations = len(self.unique_words)
        self.n_hidden_states = len(self.unique_pos)

        self.initial_probabilities = np.zeros(self.n_hidden_states)
        for line in sentences:
            first_pair = line[0]
            self.initial_probabilities[self.pos_to_idx[first_pair[1]]] += 1
        self.initial_probabilities /= self.initial_probabilities.sum()
        self.initial_probabilities = smoothen(self.initial_probabilities, smoothing)

        self.transition_probabilities = np.zeros((self.n_hidden_states, self.n_hidden_states))
        for line in sentences:
            for first, second in zip(line[:-1], line[1:]):
                self.transition_probabilities[self.pos_to_idx[first[1]], self.pos_to_idx[second[1]]] += 1
        self.transition_probabilities /= np.sum(self.transition_probabilities, axis=1, keepdims=True)
        self.transition_probabilities[np.isnan(self.transition_probabilities)] = 1 / self.transition_probabilities.shape[1]
        self.transition_probabilities = smoothen(self.transition_probabilities, smoothing)

        self.emission_probabilities = np.zeros((self.n_hidden_states, self.n_observations))
        for pair in list_of_all_pairs:
            self.emission_probabilities[self.pos_to_idx[pair[1]], self.word_to_idx[pair[0]]] += 1
        self.emission_probabilities /= np.sum(self.emission_probabilities, axis=1, keepdims=True)
        self.emission_probabilities[np.isnan(self.emission_probabilities)] = 1 / self.emission_probabilities.shape[1]
        self.emission_probabilities = smoothen(self.emission_probabilities, smoothing)

    def predict(self, sentence):
        """
        :param sentence: list of words
        :return: list of pos tags
        """
        if self.algorithm == "exhaustive":
            return self._exhaustive(sentence)
        elif self.algorithm == "viterbi":
            return self._viterbi(sentence)
        raise ValueError("Unknown algorithm: " + self.algorithm)

    def _exhaustive(self, sentence):

        sentence = [word if word in self.unique_words else "UNKNOWN" for word in sentence]

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

            if probability >= best_probability:
                best_probability = probability
                best_combination = combination

        return tuple(self.idx_to_pos[idx] for idx in best_combination)

    def _viterbi(self, sentence):
        sentence = [word if word in self.unique_words else "UNKNOWN" for word in sentence]

        T = len(sentence)  # total no. of words in the sentence
        S = self.n_hidden_states  # total no. of hidden states or the PoS tags

        if T == 0:
            return ()

        dp = np.full((T,S), -np.inf)
        backpointer = np.zeros((T,S), dtype=int)

        log_init = np.log(self.initial_probabilities)
        log_trans = np.log(self.transition_probabilities)
        log_emit = np.log(self.emission_probabilities)

        first_word_idx = self.word_to_idx[sentence[0]]

        for s in range(S):
            dp[0, s] = log_init[s] + log_emit[s, first_word_idx]

        for t in range(1, T):
            word_idx = self.word_to_idx[sentence[t]]

            scores = dp[t-1][:, None] + log_trans

            best_prev = np.argmax(scores, axis=0)

            dp[t] = scores[best_prev, range(S)] + log_emit[:, word_idx]

            backpointer[t] = best_prev

        best_last = np.argmax(dp[T-1])

        best_path = [best_last]

        for t in range(T-1, 0, -1):
            best_last = backpointer[t, best_last]
            best_path.append(best_last)
        
        best_path.reverse()

        return tuple(self.idx_to_pos[idx] for idx in best_path)

    def score(self, sentences, scorer=micro_accuracy_score):
        """
        :param sentences: A single sentence or a list of sentences, each sentence is a list of (word, pos) pairs
        :param scorer: scorer
        :return: accuracy
        """
        if not isinstance(sentences[0], list): sentences = [sentences]
        ls_wrd = [[word for word, pos in sentence] for sentence in sentences]
        ls_pos = [[pos for word, pos in sentence] for sentence in sentences]
        y_true = [pos for sentence in ls_pos for pos in sentence]
        y_pred = [pos for sentence in ls_wrd for pos in self.predict(sentence)]
        return scorer(y_true, y_pred)
