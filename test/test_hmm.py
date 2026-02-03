import numpy as np
# from hmm_viterbi import HiddenMarkovModel, macro_accuracy_score

from dummy_data import synthetic_corpus, big_synthetic_corpus

from hmm_viterbi.hmm import HiddenMarkovModel
from hmm_viterbi.scoring import macro_accuracy_score


def test_model_sanity():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive')

    assert model.algorithm == 'exhaustive'
    assert model.unique_words == ['UNKNOWN', 'ka', 'lom', 'nek', 'pul', 'sen', 'ti', 'zar']
    assert model.unique_pos == ['A', 'B', 'C', 'D']
    assert model.pos_to_idx == {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    assert model.word_to_idx == {'UNKNOWN': 0, 'ka': 1, 'lom': 2, 'nek': 3, 'pul': 4, 'sen': 5, 'ti': 6, 'zar': 7}
    assert model.n_observations == 8
    assert model.n_hidden_states == 4

    assert np.isclose(model.initial_probabilities.sum(), 1.0)
    assert (np.isclose(model.transition_probabilities.sum(axis=1), 1.0)).all()
    assert (np.isclose(model.emission_probabilities.sum(axis=1), 1.0)).all()


def test_exhaustive_algorithm():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive')

    assert model.predict(['ti', 'nek', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_viterbi_algorithm():
    model = HiddenMarkovModel(synthetic_corpus)

    assert model.algorithm == 'viterbi'
    assert model.predict(['ti', 'nek', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_unknown_handling():
    model = HiddenMarkovModel(big_synthetic_corpus, algorithm='exhaustive')

    assert model.predict(['ti', 'vib', 'ruk', 'mu']) == ('A', 'B', 'C', 'D')
    assert model.predict(['zo', 'glorp', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_scoring():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive')

    assert model.score([('ti', 'A'), ('nek', 'B'), ('pul', 'C'), ('fi', 'D')]) == 1.0
    assert model.score([('ti', 'A'), ('nek', 'B'), ('pul', 'C'), ('fi', 'A')]) == 0.75
    assert model.score([('ti', 'A'), ('nek', 'B'), ('pul', 'C'), ('fi', 'A')], scorer=macro_accuracy_score) == 5 / 6
