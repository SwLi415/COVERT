import copy
import torch
import random

poison_pattern = {'0': [[0, 0], [0, 1], [0, 2], [0, 3]],
                  '1': [[1, 4], [1, 5], [1, 6], [1, 7]],
                  '2': [[2, 6], [2, 7], [2, 8], [2, 9]]}


def add_trigger(image, adversarial_index, dataset, trigger_value):
    new_image = image.clone() if torch.is_tensor(image) else copy.deepcopy(image)
    if adversarial_index == -1:
        for i in range(0, len(poison_pattern)):
            poi_patterns = poison_pattern[str(i)]
            for j in range(0, len(poi_patterns)):
                pos = poi_patterns[j]
                new_image[i][pos[0]][pos[1]] = trigger_value[i][pos[0]][pos[1]]
    else:
        poi_patterns = poison_pattern[str(adversarial_index)]
        for i in range(0, len(poi_patterns)):
            pos = poi_patterns[i]
            new_image[adversarial_index][pos[0]][pos[1]] = trigger_value[adversarial_index][pos[0]][pos[1]]

    return new_image


def add_trigger_batch(images, adversarial_index, dataset, trigger_value, limit=None):
    new_images = images.clone()
    batch_size = new_images.size(0)
    poison_count = batch_size if limit is None else min(limit, batch_size)
    if poison_count <= 0:
        return new_images

    target_images = new_images[:poison_count]
    trigger_value = trigger_value.to(target_images.device)

    if adversarial_index == -1:
        for channel in range(len(poison_pattern)):
            poi_patterns = poison_pattern[str(channel)]
            rows = [pos[0] for pos in poi_patterns]
            cols = [pos[1] for pos in poi_patterns]
            target_images[:, channel, rows, cols] = trigger_value[channel, rows, cols]
    else:
        poi_patterns = poison_pattern[str(adversarial_index)]
        rows = [pos[0] for pos in poi_patterns]
        cols = [pos[1] for pos in poi_patterns]
        target_images[:, adversarial_index, rows, cols] = trigger_value[adversarial_index, rows, cols]

    return new_images

