import numpy as np
# from hmm_viterbi import HiddenMarkovModel, macro_accuracy_score

from dummy_data import synthetic_corpus, big_synthetic_corpus, count_unique_words

from hmm_viterbi.hmm import HiddenMarkovModel
from hmm_viterbi.scoring import macro_accuracy_score

from hmm_viterbi.hmm import simple_unknown_imputer, extensive_unknown_imputer


def test_model_sanity():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive', rarity_factor=0.7)

    assert model.algorithm == 'exhaustive'
    assert model.unique_words == ['<UNKNOWN>', 'ka', 'lom', 'pul', 'sen', 'ti', 'zar']
    assert model.unique_pos == ['A', 'B', 'C', 'D']
    assert model.pos_to_idx == {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    assert model.word_to_idx == {'<UNKNOWN>': 0, 'ka': 1, 'lom': 2, 'pul': 3, 'sen': 4, 'ti': 5, 'zar': 6}
    assert model.n_observations == 7
    assert model.n_hidden_states == 4

    assert np.isclose(model.initial_probabilities.sum(), 1.0)
    assert (np.isclose(model.transition_probabilities.sum(axis=1), 1.0)).all()
    assert (np.isclose(model.emission_probabilities.sum(axis=1), 1.0)).all()


def test_exhaustive_algorithm():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive', rarity_factor=0.7)

    assert model.predict(['ti', 'nek', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_extensive_imputation_in_hmm():
    model = HiddenMarkovModel(big_synthetic_corpus, algorithm='exhaustive', rarity_factor=0.7,
                              imputer=extensive_unknown_imputer)

    assert model.predict(['zo', 'glorp', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_viterbi_algorithm():
    model = HiddenMarkovModel(synthetic_corpus, rarity_factor=0.7)

    assert model.algorithm == 'viterbi'
    assert model.predict(['ti', 'vib', 'ruk', 'mu']) == ('A', 'B', 'C', 'D')
    assert model.predict(['ti', 'nek', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_unknown_handling():
    model = HiddenMarkovModel(big_synthetic_corpus, algorithm='exhaustive', rarity_factor=0.7)

    assert model.predict(['ti', 'vib', 'ruk', 'mu']) == ('A', 'B', 'C', 'D')
    assert model.predict(['zo', 'glorp', 'pul', 'fi']) == ('A', 'B', 'C', 'D')


def test_scoring():
    model = HiddenMarkovModel(synthetic_corpus, algorithm='exhaustive', rarity_factor=0.7)

    assert model.score([('ti', 'A'), ('nek', 'B'), ('pul', 'C'), ('fi', 'D')]) == 1.0
    assert model.score([('ti', 'A'), ('nek', 'B'), ('pul', 'C'), ('fi', 'A')]) == 0.75
    assert model.score([('ti', 'A'), ('nek', 'B'), ('pul', 'C'), ('fi', 'A')], scorer=macro_accuracy_score) == 5 / 6


def test_rarity_factor():
    model = HiddenMarkovModel(synthetic_corpus, rarity_factor=0.0)
    n_actual_words = count_unique_words(synthetic_corpus)
    assert model.n_observations == n_actual_words
    model = HiddenMarkovModel(synthetic_corpus, rarity_factor=1.0)
    assert model.n_observations - 1 == int(n_actual_words ** 0.5)  # -1 for UNKNOWN


def test_smoothing_factor():
    model = HiddenMarkovModel(synthetic_corpus, smoothing=0.0)
    n_actual_words = count_unique_words(synthetic_corpus)
    assert model.n_observations == n_actual_words
    model = HiddenMarkovModel(synthetic_corpus, rarity_factor=1.0)
    assert model.n_observations - 1 == int(n_actual_words ** 0.5)  # -1 for UNKNOWN


def test_simple_unknown_imputer():
    knowledge = {'ka', 'zar'}
    assert simple_unknown_imputer("ka", knowledge) == "ka"
    assert simple_unknown_imputer("ta", knowledge) == "<UNKNOWN>"
    assert simple_unknown_imputer(("ka", "PRONOUN"), knowledge) == ("ka", "PRONOUN")
    assert simple_unknown_imputer(("ta", "NOUN"), knowledge) == ("<UNKNOWN>", "NOUN")
    assert simple_unknown_imputer(["ka", "ta"], knowledge) == ["ka", "<UNKNOWN>"]
    assert simple_unknown_imputer([("ka", "PRONOUN"), ("ta", "NOUN")], knowledge) == [("ka", "PRONOUN"), ("<UNKNOWN>", "NOUN")]
    assert simple_unknown_imputer([[("ka", "PRONOUN"), ("ta", "NOUN")], [("ta", "NOUN"), ("ka", "PRONOUN")]], knowledge) == [[("ka", "PRONOUN"), ("<UNKNOWN>", "NOUN")], [("<UNKNOWN>", "NOUN"), ("ka", "PRONOUN")]]

def test_extensive_unknown_imputer():
    knowledge = {'ka', 'zar'}
    assert extensive_unknown_imputer(("ka", "PRONOUN"), knowledge) == ("ka", "PRONOUN")
    assert extensive_unknown_imputer(("ta", "NOUN"), knowledge) == ("<UNKNOWN:NOUN>", "NOUN")
    assert extensive_unknown_imputer([("ka", "PRONOUN"), ("ta", "NOUN")], knowledge) == [("ka", "PRONOUN"), ("<UNKNOWN:NOUN>", "NOUN")]
    assert extensive_unknown_imputer([[("ka", "PRONOUN"), ("ta", "NOUN")], [("ta", "NOUN"), ("ka", "PRONOUN")]], knowledge) == [[("ka", "PRONOUN"), ("<UNKNOWN:NOUN>", "NOUN")], [("<UNKNOWN:NOUN>", "NOUN"), ("ka", "PRONOUN")]]
