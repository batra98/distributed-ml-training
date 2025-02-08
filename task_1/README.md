# Part 1: Training VGG-11 on CIFAR-10

In this part, we trained the VGG-11 model on the CIFAR-10 dataset for one epoch with a batch size of 256. The training was performed over 196 iterations for the full dataset and 40 iterations for a smaller subset. Below are the detailed results for both cases:

## Metrics After Running the CNN for One Epoch (Batch Size: 256, 196 Iterations)

| Epoch | Iteration | Loss   |
| ----- | --------- | ------ |
| 1     | 20/196    | 3.3479 |
| 1     | 40/196    | 2.5626 |
| 1     | 60/196    | 2.2670 |
| 1     | 80/196    | 2.3202 |
| 1     | 100/196   | 2.2707 |
| 1     | 120/196   | 2.2434 |
| 1     | 140/196   | 2.2375 |
| 1     | 160/196   | 2.2207 |
| 1     | 180/196   | 2.2134 |

- **Training Summary**:
  - **Average Loss (Training)**: 2.7683
  - **Training Accuracy**: 12.40%
  - **Test Accuracy**: 17.00%
  - **Average Time per Iteration** (after discarding the first iteration): 2.1528 seconds

---

## Metrics After Running the CNN for One Epoch (Batch Size: 256, 40 Iterations)

| Epoch | Iteration | Loss   |
| ----- | --------- | ------ |
| 1     | 20/40     | 2.8673 |
| 1     | 40/40     | 2.3245 |

- **Training Summary**:
  - **Average Loss (Training)**: 0.9930
  - **Training Accuracy**: 11.51%
  - **Average Time per Iteration** (after discarding the first iteration): 2.3370 seconds

---
