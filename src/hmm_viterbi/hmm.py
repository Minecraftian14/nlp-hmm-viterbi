import numpy as np
import itertools as it
import collections as cls

from .scoring import micro_accuracy_score


def simple_unknown_imputer(data, knowledge: set[str], insight: any = None):
    if isinstance(data, str):  # is a word
        return data if data in knowledge else "<UNKNOWN>"
    if isinstance(data, tuple):  # is a pair
        return data[0] if data[0] in knowledge else "<UNKNOWN>", data[1]
    if isinstance(data, list):
        return [simple_unknown_imputer(x, knowledge) for x in data]
    raise ValueError("Unknown data type: " + str(type(data)))


def extensive_unknown_imputer(data, knowledge: set[str], insight: 'HiddenMarkovModel' = None):
    if isinstance(data, str):  # is a word
        return data if data in knowledge else f"<UNKNOWN>"
    if isinstance(data, tuple):  # is a pair
        return data[0] if data[0] in knowledge else f"<UNKNOWN:{data[1]}>", data[1]
    if isinstance(data, list):
        data = [extensive_unknown_imputer(x, knowledge, insight) for x in data]

        if isinstance(data[0], str):  # is a word

            if data[0] == "<UNKNOWN>":
                first_pos = insight.initial_probabilities.argmax()
                data[0] = f"<UNKNOWN:{insight.idx_to_pos[first_pos]}>"

            first = data[0]
            for index, second in enumerate(data[1:]):
                if second == "<UNKNOWN>":
                    # Based on the first word, what's the most likely POS is it itself?
                    likely_pos = insight.emission_probabilities[:, insight.word_to_idx[first]].argmax()
                    # Based on the first pos, what's the most likely POS is second itself?
                    likely_pos = insight.transition_probabilities[likely_pos, :].argmax()

                    data[index + 1] = second = f"<UNKNOWN:{insight.idx_to_pos[likely_pos]}>"

                first = second

        return data
    raise ValueError("Unknown data type: " + str(type(data)))


def smoothen(matrix, smoothing):
    return matrix * (1 - smoothing) + smoothing / matrix.shape[-1]


class HiddenMarkovModel:

    def __init__(self,
                 sentences: list[list[tuple[str, str]]],
                 algorithm: str = 'viterbi',
                 smoothing: float = 0.01,
                 rarity_factor: float = 0.05,
                 imputer=simple_unknown_imputer,
                 ignore_rarity_guard_rail=False,
                 ):
        """
        :param sentences: List of (word, pos) pairs
        :param algorithm: 'exhaustive' or 'viterbi'
        :param smoothing: Smoothing parameter (interpolation between actual and uniform probabilities). 1 means use actual probabilities; 0 means use uniform probabilities.
        :param rarity_factor: Threshold for rarity of words in the corpus. 0 means all words retained; 1 means all words discarded. There is a hard constraint to retain at least len(counter)**0.5 words.
        :param imputer: The unknown word handling mechanism.
        :param ignore_rarity_guard_rail: Analytics helper for rarity_factor. Disables the minimum len(counter)**0.5 criteria.
        """

        assert len(sentences) > 0, ""
        assert algorithm in ['exhaustive', 'viterbi'], ""
        assert 0 <= smoothing < 1, ""
        assert 0 <= rarity_factor <= 1, ""

        self.algorithm = algorithm
        self.smoothing = smoothing
        self.rarity_factor = rarity_factor
        self.imputer = imputer

        # Flatten the list for easier processing
        list_of_all_pairs = [pair for line in sentences for pair in line]

        # Create a counter for calculating the frequencies
        counter = cls.Counter([word for word, _ in list_of_all_pairs])
        # Pre-Calculate the Offset Log of the most common frequency for filtering
        max_frequency_factor = np.log(counter.most_common()[0][1] + 1) * rarity_factor
        # Calculate a retention factor to keep only the most common words:
        #   Either keep 100*rarity_factor% of the most common words by logscale
        #   Or at least len(all words) ** 0.5
        # Any values that occur less than 100rarity_factor% in the corpus
        low_rankers = [x for x in counter.values() if np.log(x + 1) <= max_frequency_factor]
        if len(low_rankers) == 0: low_rankers = [counter.most_common()[-1][1]]
        side_rank = len([x for x in counter.values() if x < max(low_rankers)])
        threshold_rank = int(side_rank + (len(low_rankers) - side_rank) * rarity_factor)
        # At least pick up some words
        if not ignore_rarity_guard_rail: threshold_rank = min(threshold_rank, int(len(counter) - len(counter) ** 0.5))
        # Use the retention factor to filter out rare words
        self.knowledge = {word for word, freq in counter.most_common()[:(len(counter) - threshold_rank)]}

        # Replace excluded words with "UNKNOWN"
        # list_of_all_pairs = [
        #     ((word, pos) if word in knowledge else ("UNKNOWN", pos))
        #     for word, pos in list_of_all_pairs]
        list_of_all_pairs = imputer(list_of_all_pairs, self.knowledge)

        # sentences = [
        #     [
        #         ((word, pos) if word in knowledge else ("UNKNOWN", pos))
        #         for (word, pos) in sentence]
        #     for sentence in sentences]
        sentences = imputer(sentences, self.knowledge)

        # Finalize the HMM architecture
        # *knowledge is what words were known and unique_words is all known words + unknown placeholders*
        self.unique_pos = sorted(list({pos for word, pos in list_of_all_pairs}))
        self.n_hidden_states = len(self.unique_pos)
        self.unique_words = {word for word, pos in list_of_all_pairs}
        self.unique_words |= {x[0] for x in (self.imputer([(x, x) for x in self.unique_pos], set()))}
        self.unique_words = sorted(list(self.unique_words))
        self.n_observations = len(self.unique_words)

        # Create some helpers for index<->token conversion
        self.pos_to_idx = {pos: idx for idx, pos in enumerate(self.unique_pos)}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.unique_words)}
        self.idx_to_pos = {v: k for k, v in self.pos_to_idx.items()}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

        # Calculate the initial probabilities
        self.initial_probabilities = np.zeros(self.n_hidden_states)
        for line in sentences:
            first_pair = line[0]
            self.initial_probabilities[self.pos_to_idx[first_pair[1]]] += 1
        self.initial_probabilities /= self.initial_probabilities.sum()
        self.initial_probabilities = smoothen(self.initial_probabilities, smoothing)

        # Calculate the transition probabilities
        self.transition_probabilities = np.zeros((self.n_hidden_states, self.n_hidden_states))
        for line in sentences:
            for first, second in zip(line[:-1], line[1:]):
                self.transition_probabilities[self.pos_to_idx[first[1]], self.pos_to_idx[second[1]]] += 1
        self.transition_probabilities /= np.sum(self.transition_probabilities, axis=1, keepdims=True)
        self.transition_probabilities[np.isnan(self.transition_probabilities)] = 1 / self.transition_probabilities.shape[1]
        self.transition_probabilities = smoothen(self.transition_probabilities, smoothing)

        # Calculate the emission probabilities
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
        sentence = self.imputer(sentence, self.knowledge, self)

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

            if probability >= best_probability:
                best_probability = probability
                best_combination = combination

        return tuple(self.idx_to_pos[idx] for idx in best_combination)

    def _viterbi(self, sentence):

        T = len(sentence)  # total no. of words in the sentence
        S = self.n_hidden_states  # total no. of hidden states or the PoS tags

        if T == 0:
            return ()

        dp = np.full((T, S), -np.inf)
        backpointer = np.zeros((T, S), dtype=int)

        log_init = np.log(self.initial_probabilities)
        log_trans = np.log(self.transition_probabilities)
        log_emit = np.log(self.emission_probabilities)

        first_word_idx = self.word_to_idx[sentence[0]]

        for s in range(S):
            dp[0, s] = log_init[s] + log_emit[s, first_word_idx]

        for t in range(1, T):
            word_idx = self.word_to_idx[sentence[t]]

            scores = dp[t - 1][:, None] + log_trans

            best_prev = np.argmax(scores, axis=0)

            dp[t] = scores[best_prev, range(S)] + log_emit[:, word_idx]

            backpointer[t] = best_prev

        best_last = np.argmax(dp[T - 1])

        best_path = [best_last]

        for t in range(T - 1, 0, -1):
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
