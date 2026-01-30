from hmm_viterbi import *


def test_micro_accuracy():
    assert micro_accuracy_score(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D']) == 1.0
    assert micro_accuracy_score(['A', 'B', 'C', 'D'], ['A', 'B', 'X', 'Y']) == 0.5


def test_macro_accuracy():
    assert macro_accuracy_score([
        'A', 'A',
        'B',
        'C', 'C', 'C'], [
        'A', 'B',
        'B',
        'C', 'C', 'D']) == 13 / 18


def test_confusion_matrix():
    cm = confusion_matrix([
        'A', 'A',
        'B',
        'C', 'C', 'C'], [
        'A', 'B',
        'B',
        'C', 'C', 'D'])
    assert cm == {'A': {'A': 1, 'B': 1, 'C': 0}, 'B': {'B': 1, 'A': 0, 'C': 0}, 'C': {'C': 2, 'D': 1, 'A': 0, 'B': 0}}
    display_confusion_matrix(cm)
