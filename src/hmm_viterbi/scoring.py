def micro_accuracy_score(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def macro_accuracy_score(y_true, y_pred):
    scores = {}
    for t, p in zip(y_true, y_pred):
        if t not in scores: scores[t] = [0, 0]
        scores[t][0] += t == p
        scores[t][1] += 1
    for key, (count, total) in scores.items():
        scores[key] = count / total
    return sum(scores.values()) / len(scores)


def confusion_matrix(y_true, y_pred):
    matrix = {}  # dynamically updated
    for t, p in zip(y_true, y_pred):
        if t not in matrix: matrix[t] = {}
        if p not in matrix[t]: matrix[t][p] = 0
        matrix[t][p] += 1
    for col in matrix.values():
        for key in matrix:
            if key not in col: col[key] = 0
    return matrix


def display_confusion_matrix(matrix):
    matrix = {t: {p: str(v) for p, v in c.items()} for t, c in matrix.items()}
    keys = sorted(map(str, matrix.keys()))
    width = max(map(len, keys))
    for col in matrix.values():
        width = max(width, max(map(len, col.values())))
    width += 1
    print(' ' * width, end='')
    for key in keys:
        print(f'{key:>{width}}'.ljust(width), end='')
    print()
    for t in keys:
        print(f'{t:>{width}}', end='')
        for p in keys:
            print(f'{matrix[t][p]:>{width}}', end='')
        print()
