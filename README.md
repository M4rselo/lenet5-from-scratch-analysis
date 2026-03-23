# ***LeNet-5 from Scratch: Analysis on MNIST and CIFAR-10***
This project implements LeNet-5 from scratch in PyTorch using custom-built layers, and analyzes model's behavior across two datasets of varying complexity: MNIST and CIFAR-10.
Analysis focuses on differences in generalization behavior, the impact of pooling operations, and feature map visualization of patterns learned by convolutional layers.
Additionally, misclassified examples are examined to identify common error patterns and highlight weaknesses in the model’s predictions.

## Architecture
- LeNet-5 architecture composed of *convolutional, pooling, and fully connected layers*
- Standard linear classifier with cross-entropy loss (instead of original gaussian connections)
- ReLU activations (instead of original tanh)
- Non-trainable pooling layers (instead of original sub-sampling trainable layers)
- Support for both max and average pooling

## Implementation
- Custom convolutional and linear layers (implemented without high-level APIs such as nn.Conv2d / nn.Linear)
- Vectorized sliding-window convolution approach
- Custom pooling layer supporting max and average operations
- He initialization for convolutional and linear layers
- Convolution → pooling → fully connected pipeline with ReLU activations
- Custom training pipeline with dedicated Trainer and Plotter classes, including manual batch step (forward pass, loss, accuracy)
- Stochastic Gradient Descent (SGD) optimizer
- Explicit device management (CPU/GPU support)

## Experiments
- Model perfomance comparison between MNIST and CIFAR-10
- Comparison of average vs max pooling
- Analysis of generelization gap and overfitting tendencies
- Analysis of model's limited capacity and its on performance 

## Visualization
- Dataset samples before training 
- Feature maps from convolutional layers
- Analysis of misclassified examples and error patterns

## Key Conclusions
- LeNet-5 performs exceptionally well on MNIST but struggles on CIFAR-10, highlighting the limitations of its capacity when handling more complex, high-dimensional data.
- Max pooling has minimal impact on accuracy compared to average pooling, though it slightly reduces the generalization gap on CIFAR-10.
- Feature map visualization highlights the filters' consistent ability to distinguish specific features across different inputs, resulting in similar responses within each filter, especially visible on the MNIST dataset.
- An analysis of misclassified examples reveals distinct patterns: CIFAR-10 errors are mostly caused by visually similar classes and shared background characteristics, while MNIST misclassifications mostly come from ambiguous or poorly written digits.

*Full, detailed conclusions are available at the end of the notebook.*

## Notebook
`lenet5_from_scratch_analysis.ipynb`
