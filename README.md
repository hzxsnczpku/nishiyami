# A Code Implementation for the Statoil/C-CORE Iceberg Classifier Challenge

Here is my code implementation of the Statoil/C-CORE Iceberg Classifier Challenge.

## Implemented Modules
### Traditional Models

* Supporter Vector Machine

### Neural Networks

* LeNet
* VGG
  * VGG16
  * VGG19
* ResNet
* DenseNet
* SE-ResNet

### Filters
* PCA Filter
* Lee Filter

### Ensemble Methods
* Mean
* Medium
* Pushout
* MinMax
* BestBase
* Denoising


## Experiment Results
### SVM
#### Basic

| model    | train loss | train acc | val loss| val acc |
|:--------:|:----------:|:---------:|:-------:|:-------:|
| Gaussian | 0.1053     | 1.0       | 0.6746  | 0.5860  |
| linear   | 0.3058     | 1.0       | 0.5818  | 0.6920  |

#### PCA: k=20

| model    | train loss | train acc | val loss| val acc |
|:--------:|:----------:|:---------:|:-------:|:-------:|
| Gaussian | 0.5758     | 0.7076    | 0.5911  | 0.7007  |
| linear   | 0.5670     | 0.7138    | 0.5843  | 0.6926  |

#### PCA: k=100

| model    | train loss | train acc | val loss| val acc |
|:--------:|:----------:|:---------:|:-------:|:-------:|
| Gaussian | 0.5338     | 0.7344    | 0.5839  | 0.6951  |
| linear   | 0.5295     | 0.7506    | 0.5745  | 0.6970  |

#### PCA: k=200

| model    | train loss | train acc | val loss| val acc |
|:--------:|:----------:|:---------:|:-------:|:-------:|
| Gaussian | 0.5174     | 0.7774    | 0.5904  | 0.6833  |
| linear   | 0.5173     | 0.7737    | 0.5940  | 0.6802  |

### NN
#### Basic
Under Construction
