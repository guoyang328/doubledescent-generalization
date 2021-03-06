# Optimization Variance: Delve into the Epoch-Wise Double Descent of DNNs

Exploring the implementation **Optimization Variance: Delve into the Epoch-Wise Double Descent of DNNs** from https://github.com/ZhangXiao96/OptimizationVariance

## Requirements

pytorch = 1.5.1
tensorboardX

## Training

To train the model in the paper, run this command:

```train
python train.py data_name model_name noise_split opt lr test_id data_root
# data_name: 'svhn', 'cifar10', 'cifar100'
# model_name: 'vgg11', 'vgg13', 'vgg16', 'resnet18', 'resnet34'
# noise_split: [0, 1)
# opt: 'adam', 'sgd'
# lr: learning rate
# test_id: 0, marker
# data_root: direction of datasets
```

## Plot Optimization Variance and Acc

see log_to_csv.py and plot_var_acc.py

## Results

OV and ACC:
![](assets/example.png)

