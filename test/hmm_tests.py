from hmm_viterbi import HiddenMarkovModel

from dummy_data import synthetic_corpus


def test_model_sanity():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive')

    assert model.algorithm == 'exhaustive'
    assert model.unique_words == ['fi', 'ka', 'lom', 'mu', 'nek', 'pul', 'sen', 'ti', 'zar']
    assert model.unique_pos == ['A', 'B', 'C', 'D']
    assert model.pos_to_idx == {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    assert model.word_to_idx == {'fi': 0, 'ka': 1, 'lom': 2, 'mu': 3, 'nek': 4, 'pul': 5, 'sen': 6, 'ti': 7, 'zar': 8}
    assert model.n_observations == 9
    assert model.n_hidden_states == 4

    assert model.initial_probabilities.sum() == 1.0
    assert (model.transition_probabilities.sum(axis=1) == 1.0).all()
    assert (model.emission_probabilities.sum(axis=1) == 1.0).all()


def test_exhaustive_algorithm():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive')

    assert model.predict(['ti', 'nek', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_viterbi_algorithm():
    model = HiddenMarkovModel(synthetic_corpus)

    assert model.algorithm == 'viterbi'
    assert model.predict(['ti', 'nek', 'pul', 'fi']) == ('A', 'B', 'C', 'D')
