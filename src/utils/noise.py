import numpy as np
import torch


def mnist_structured_noise(dataset, probability, digits, replacements):
    # Check that probability is a float in the range [0_baseline, 1]
    if not isinstance(probability, float) or probability < 0 or probability > 1:
        raise ValueError("Invalid probability")

    # Check that digits and replacements are lists of the same length
    if not isinstance(digits, list) or not isinstance(replacements, list) or len(digits) != len(replacements):
        raise ValueError("Invalid digits or replacements")

    # Check that digits and replacements only contain valid digits
    for digit in digits:
        if not isinstance(digit, int) or digit < 0 or digit > 9:
            raise ValueError("Invalid digit in digits")
    for replacement in replacements:
        if not isinstance(replacement, int) or replacement < 0 or replacement > 9:
            raise ValueError("Invalid digit in replacements")

    # Create a mask to store the digits that should be flipped
    mask = torch.zeros(len(dataset), dtype=torch.bool)
    # Iterate over the dataset and flip the labels of the selected digits
    for i, (data, label) in enumerate(dataset):
        if label in digits:
            if torch.rand(1) < probability:
                # Flip the label to the corresponding replacement
                index = digits.index(label)
                dataset[i] = (data, replacements[index])
                mask[i] = True

    # Return the modified dataset and the mask of flipped examples
    return dataset, mask


def mnist_label_noise(dataset, pc_corrupt):
    # Check that probability is a float in the range [0_baseline, 1]
    if not isinstance(pc_corrupt, float) or pc_corrupt < 0 or pc_corrupt > 1:
        raise ValueError("Invalid probability")

    number_points = len(dataset)
    # Create a mask to store the digits that should be flipped
    mask = torch.zeros(len(dataset), dtype=torch.bool)

    selected_indecies = np.random.choice(np.arange(0, number_points), int(number_points * pc_corrupt), replace=False)
    # Iterate over the dataset and flip the labels of the selected indices
    for index in selected_indecies:
        data, _target = dataset[index]
        dataset[index] = (data, np.random.randint(0, 9))
        mask[index] = True

    # Return the modified dataset and the mask of flipped examples
    return dataset, mask


def get_num_classes(dataloader):
    num_classes = 0
    for data, label in dataloader:
        # Determine the number of classes by taking the maximum value in the labels tensor
        # plus one (since classes are typically numbered starting from zero)
        num_classes = max(num_classes, torch.max(label).item() + 1)

    return num_classes
