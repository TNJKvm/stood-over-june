import random
import numpy as np


def unbalance_dataset(dataset, classes=None):
    num_classes = 20
    selected_classes = random.sample(list(dataset.class_to_idx.values()), num_classes) if classes is None else classes

    labels = np.array(dataset.targets)
    images = np.array(dataset.data)

    indices_selected = np.argwhere(np.isin(labels, selected_classes)).flatten()
    c = 0
    for i in range(100):
        if i not in selected_classes:
            label_indices = np.argwhere(labels == i).flatten()
            indices_selected = np.concatenate((indices_selected, np.random.choice(label_indices, size=int(len(label_indices) * 0.06))))
        else:
            c += 1
    return list(images[indices_selected]), list(labels[indices_selected]), selected_classes
