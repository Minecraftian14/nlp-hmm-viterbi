import numpy as np
import matplotlib.pyplot as plt


def count_unique_words(corpus: list[list[tuple[str, str]]]):
    return len({word for sentence in corpus for word, pos in sentence})


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


def display_confusion_matrix_heatmap(matrix, normalize=True, title="Confusion Matrix"):
    tags = sorted(matrix.keys())

    cm = np.array([[matrix[t][p] for p in tags] for t in tags], dtype=float)

    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(tags)))
    ax.set_yticks(np.arange(len(tags)))
    ax.set_xticklabels(tags)
    ax.set_yticklabels(tags)

    plt.xticks(rotation=45)

    for i in range(len(tags)):
        for j in range(len(tags)):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else int(val)
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
