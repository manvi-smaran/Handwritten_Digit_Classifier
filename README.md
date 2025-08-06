

#  MNIST Handwritten Digit Classifier

This project involves building and evaluating a neural network using **PyTorch** to classify handwritten digits from the **MNIST** dataset. The work is divided into three roles:  
🔹 **Baseline Modeling**  
🔹 **Hyperparameter Experimentation**  
🔹 **Analysis & Reporting**
---

## 🚀 Best Performing Model

| Parameter          | Value               |
|-------------------|---------------------|
| Hidden Layers      | 2                   |
| Neurons per Layer | 128                 |
| Activation        | ReLU                |
| Optimizer         | Adam                |
| Epochs            | 20                  |
| **Test Accuracy** | **98.2%** (approx)  |


<br>

##  Model Architecture

```python
nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```



<br>

# Observation Table
| **Config** | **Epochs** |              **Layers**             |              **Neurons**             | **Activation** | **Optimizer** | **Test Accuracy (%)** |
|:----------:|:----------:|:-----------------------------------:|:------------------------------------:|:--------------:|:-------------:|:---------------------:|
|     #1     |      5     |           1 (Output Layer)          |                  128                 |      relu      |      adam     |         97.33         |
|     #2     |      5     |  2 (1 Output Layer; 1 Hidden Layer) |      256 (L1 -> 256; L2 -> 128)      |      relu      |      adam     |         97.84         |
|     #3     |      5     | 3 (1 Output Layer; 2 Hidden Layers) | 256 (L1 -> 256; L2 -> 128; L3 -> 64) |      relu      |      adam     |         97.60         |
|     #4     |      5     |           1 (Output Layer)          |                  128                 |      tanh      |      sgd      |         95.10         |
|     #5     |      5     |  2 (1 Output Layer; 1 Hidden Layer) |      256 (L1 -> 256; L2 -> 128)      |      tanh      |      sgd      |         95.51         |
|     #6     |      5     | 3 (1 Output Layer; 2 Hidden Layers) | 256 (L1 -> 256; L2 -> 128; L3 -> 64) |      tanh      |      sgd      |         95.79         |
|     #7     |     10     |           1 (Output Layer)          |                  128                 |      relu      |      adam     |         97.32         |
|     #8     |     10     |  2 (1 Output Layer; 1 Hidden Layer) |      256 (L1 -> 256; L2 -> 128)      |      relu      |      adam     |         97.92         |
|     #9     |     10     | 3 (1 Output Layer; 2 Hidden Layers) | 256 (L1 -> 256; L2 -> 128; L3 -> 64) |      relu      |      adam     |         97.98         |
|     #10    |     10     |           1 (Output Layer)          |                  128                 |      tanh      |      sgd      |         96.63         |
|     #11    |     10     |  2 (1 Output Layer; 1 Hidden Layer) |      256 (L1 -> 256; L2 -> 128)      |      tanh      |      sgd      |         96.94         |
|     #12    |     10     | 3 (1 Output Layer; 2 Hidden Layers) | 256 (L1 -> 256; L2 -> 128; L3 -> 64) |      tanh      |      sgd      |         97.34         |
|     #13    |     20     |           1 (Output Layer)          |                  128                 |      relu      |      adam     |         97.95         |
|     #14    |     20     |  2 (1 Output Layer; 1 Hidden Layer) |      256 (L1 -> 256; L2 -> 128)      |      relu      |      adam     |         98.11         |
|     #15    |     20     | 3 (1 Output Layer; 2 Hidden Layers) | 256 (L1 -> 256; L2 -> 128; L3 -> 64) |      relu      |      adam     |         98.00         |
|     #16    |     20     |           1 (Output Layer)          |                  128                 |      tanh      |      sgd      |         97.74         |
|     #17    |     20     |  2 (1 Output Layer; 1 Hidden Layer) |      256 (L1 -> 256; L2 -> 128)      |      tanh      |      sgd      |         98.03         |
|     #18    |     20     | 3 (1 Output Layer; 2 Hidden Layers) | 256 (L1 -> 256; L2 -> 128; L3 -> 64) |      tanh      |      sgd      |         97.99         |

<br>

## 🔍 Key Observations (based on observation table)

Across 18 experiments, three factors emerged as most influential: **activation function, optimizer, and network depth**. ReLU paired with the Adam optimizer consistently converged faster and achieved higher scores, peaking at **98.11 %** with two hidden layers and 20 epochs. While Tanh–SGD configurations improved with longer training, they generally lagged behind ReLU–Adam by ~1 % at lower epochs and matched it only after 20 epochs. Increasing layers from one to two boosted accuracy, but a third layer showed diminishing returns, suggesting the task’s complexity is adequately captured by a two‑layer architecture. Finally, larger neuron counts (256 → 128) offered modest gains, yet the lightweight 128‑neuron model still surpassed 97.9 %, highlighting a trade‑off between accuracy and computation.

<br>

## 🡩‍💻 Author

Made by [**Vemula Manvi Smaran**](https://github.com/manvi-smaran).

## 📄 License

This project is licensed under the MIT License.

