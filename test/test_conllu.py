from conllu_parser import parse_conllu_file


def test_parser():
    with open("data/UD_English-EWT/en_ewt-ud-test.conllu") as f:
        generator = parse_conllu_file(f)
        assert next(generator) == {'sentence': 'What if Google Morphed Into GoogleOS?', 'pos_tags': [('What', 'PRON'), ('if', 'SCONJ'), ('Google', 'PROPN'), ('Morphed', 'VERB'), ('Into', 'ADP'), ('GoogleOS', 'PROPN'), ('?', 'PUNCT')]}
        assert next(generator) == {'sentence': 'What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?',
                                   'pos_tags': [('What', 'PRON'), ('if', 'SCONJ'), ('Google', 'PROPN'), ('expanded', 'VERB'), ('on', 'ADP'), ('its', 'PRON'), ('search', 'NOUN'), ('-', 'PUNCT'), ('engine', 'NOUN'), ('(', 'PUNCT'),
                                                ('and', 'CCONJ'), ('now', 'ADV'), ('e-mail', 'NOUN'), (')', 'PUNCT'), ('wares', 'NOUN'), ('into', 'ADP'), ('a', 'DET'), ('full', 'ADV'), ('-', 'PUNCT'), ('fledged', 'ADJ'),
                                                ('operating', 'NOUN'), ('system', 'NOUN'), ('?', 'PUNCT')]}
