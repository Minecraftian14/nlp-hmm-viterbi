def parse_conllu_file(file):
    current_line = None
    pos_tags = []

    for line in file:
        if current_line is None:
            if line.startswith("# text = "):
                current_line = line[9:-1]
        elif line == '\n':
            yield {'sentence': current_line, 'pos_tags': pos_tags}
            current_line = None
            pos_tags = []
        else:
            desc = line.split('\t')
            index, word, pos = desc[0], desc[1], desc[3]
            if index.isnumeric():
                pos_tags.append((word, pos))
