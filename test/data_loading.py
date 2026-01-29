from conllu import parse_incr

# Path to the dataset file
path = "../data/UD_English-EWT/en_ewt-ud-test.conllu"

# Read and parse
with open(path, "r", encoding="utf-8") as f:
    sentences = list(parse_incr(f))

print(f"Loaded {len(sentences)} sentences")

# Inspect one sentence
first = sentences[0]
print(first)
