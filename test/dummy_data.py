synthetic_corpus = [
    [("ka", "A"), ("lom", "B"), ("pul", "C")],
    [("ti", "A"), ("zar", "B"), ("sen", "C")],

    [("ka", "A"), ("nek", "B"), ("pul", "C")],
    [("ti", "A"), ("lom", "B"), ("sen", "C")],

    [("ka", "A"), ("lom", "B"), ("pul", "C"), ("fi", "D")],
    [("ti", "A"), ("zar", "B"), ("sen", "C"), ("mu", "D")],

    [("ka", "A"), ("nek", "B"), ("sen", "C"), ("fi", "D")],
    [("ti", "A"), ("lom", "B"), ("pul", "C"), ("mu", "D")],

    [("lom", "B"), ("pul", "C")],
    [("zar", "B"), ("sen", "C")],
]

big_synthetic_corpus = [
    # A → B → C
    [("ka", "A"), ("lom", "B"), ("pul", "C")],
    [("ka", "A"), ("zar", "B"), ("sen", "C")],
    [("ti", "A"), ("nek", "B"), ("ruk", "C")],
    [("zo", "A"), ("vib", "B"), ("pul", "C")],

    # A → B → C → D
    [("ka", "A"), ("lom", "B"), ("pul", "C"), ("fi", "D")],
    [("ti", "A"), ("zar", "B"), ("sen", "C"), ("mu", "D")],
    [("zo", "A"), ("nek", "B"), ("ruk", "C"), ("xa", "D")],

    # B → C
    [("lom", "B"), ("pul", "C")],
    [("zar", "B"), ("sen", "C")],
    [("nek", "B"), ("ruk", "C")],

    # B → C → D
    [("vib", "B"), ("pul", "C"), ("fi", "D")],
    [("lom", "B"), ("sen", "C"), ("mu", "D")],

    # A → B → B → C (extra structure)
    [("ka", "A"), ("lom", "B"), ("zar", "B"), ("pul", "C")],
    [("ti", "A"), ("nek", "B"), ("vib", "B"), ("sen", "C")],

    # A → C (rare but allowed)
    [("ka", "A"), ("pul", "C")],
    [("zo", "A"), ("ruk", "C")],

    # Sentence starting with C
    [("pul", "C"), ("fi", "D")],
    [("sen", "C"), ("mu", "D")],

    # Sentence starting with D (rare edge case)
    [("fi", "D"), ("xa", "D")],
]
