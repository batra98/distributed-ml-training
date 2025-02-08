# Part 3: Distributed Data Parallel Training using Built in Module

---

## Metrics After Running the VGG for One Epoch (Batch Size: 256, 40 Iterations) using Distributed Data Parallel Training

### Node 0 (Rank 0)

| Epoch | Iteration | Loss   |
| ----- | --------- | ------ |
| 1     | 20/40     | 3.4271 |
| 1     | 40/40     | 2.3583 |

- **Training Summary**:
  - **Average Loss (Training)**: 3.4211
  - **Training Accuracy**: 10.26%
  - **Average Time per Iteration** (after discarding the first iteration): 3.048406 seconds

### Node 1 (Rank 1)

| Epoch | Iteration | Loss   |
| ----- | --------- | ------ |
| 1     | 20/40     | 4.3896 |
| 1     | 40/40     | 2.5432 |

- **Training Summary**:
  - **Average Loss (Training)**: 0.9930
  - **Training Accuracy**: 10.80%
  - **Average Time per Iteration** (after discarding the first iteration): 3.047395 seconds

### Node 2 (Rank 2)

| Epoch | Iteration | Loss   |
| ----- | --------- | ------ |
| 1     | 20/40     | 3.0899 |
| 1     | 40/40     | 2.6086 |

- **Training Summary**:
  - **Average Loss (Training)**: 3.3770
  - **Training Accuracy**: 10.85%
  - **Average Time per Iteration** (after discarding the first iteration): 3.048338 seconds

### Node 3 (Rank 3)

| Epoch | Iteration | Loss   |
| ----- | --------- | ------ |
| 1     | 20/40     | 3.4271 |
| 1     | 40/40     | 2.3583 |

- **Training Summary**:

  - **Average Loss (Training)**: 3.4319
  - **Training Accuracy**: 10.56%
  - **Average Time per Iteration** (after discarding the first iteration): 3.047500 seconds

- Test Summary:
  - Average loss: 2.3042
  - Accuracy: 919/10000 (9%)

---
