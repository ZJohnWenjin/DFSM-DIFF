import random

def generate_condition_idx(num_modality):
    if num_modality < 1:
        raise ValueError("num_modality must be greater than or equal to 1.")

    while True:
        condition_idx = [
            random.randint(0, 1)
            for _ in range(num_modality)
        ]

        if any(condition_idx):
            return condition_idx