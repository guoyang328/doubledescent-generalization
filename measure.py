# import torch
# import math

# # This function calculates various measures on the given model and returns a dictionary whose keys are the measure names
# # and values are the corresponding measures on the model


# def calculate(model, init_model, device, train_loader, margin):

#     # switch to evaluate mode
#     model.eval()
#     init_model.eval()

#     modules = list(model.children())
#     init_modules = list(init_model.children())

#     D = modules[0].weight.size(1)  # data dimension
#     H = modules[0].weight.size(0)  # number of hidden units
#     C = modules[2].weight.size(0)  # number of classes (output dimension)

#     # number of parameters of the model
#     num_param = sum(p.numel() for p in model.parameters())

#     with torch.no_grad():
#         # Eigenvalues of the weight matrix in the first layer
#         _, S1, _ = modules[0].weight.svd()
#         # Eigenvalues of the weight matrix in the second layer
#         _, S2, _ = modules[2].weight.svd()
#         # Eigenvalues of the initial weight matrix in the first layer
#         _, S01, _ = init_modules[0].weight.svd()
#         # Eigenvalues of the initial weight matrix in the second layer
#         _, S02, _ = init_modules[2].weight.svd()
#         # Frobenius norm of the weight matrix in the first layer
#         Fro1 = modules[0].weight.norm()
#         # Frobenius norm of the weight matrix in the second layer
#         Fro2 = modules[2].weight.norm()
#         # difference of final weights to the initial weights in the first layer
#         diff1 = modules[0].weight - init_modules[0].weight
#         # difference of final weights to the initial weights in the second layer
#         diff2 = modules[2].weight - init_modules[2].weight
#         # Euclidean distance of the weight matrix in the first layer to the initial weight matrix
#         Dist1 = diff1.norm()
#         # Euclidean distance of the weight matrix in the second layer to the initial weight matrix
#         Dist2 = diff2.norm()
#         # L_{1,infty} norm of the weight matrix in the first layer
#         L1max1 = modules[0].weight.norm(p=1, dim=1).max()
#         # L_{1,infty} norm of the weight matrix in the second layer
#         L1max2 = modules[2].weight.norm(p=1, dim=1).max()
#         # L_{2,1} distance of the weight matrix in the first layer to the initial weight matrix
#         L1Dist1 = diff1.norm(p=2, dim=1).sum()
#         # L_{2,1} distance of the weight matrix in the second layer to the initial weight matrix
#         L1Dist2 = diff2.norm(p=2, dim=1).sum()

#         measure = {}
#         measure['Frobenius1'] = Fro1
#         measure['Frobenius2'] = Fro2
#         measure['Distance1'] = Dist1
#         measure['Distance2'] = Dist2
#         measure['Spectral1'] = S1[0]
#         measure['Spectral2'] = S2[0]
#         measure['Fro_Fro'] = Fro1 * Fro2
#         measure['L1max_L1max'] = L1max1 * L1max2
#         measure['Spec_Dist'] = S1[0] * Dist2 * math.sqrt(C)
#         measure['Dist_Spec'] = S2[0] * Dist1 * math.sqrt(H)
#         measure['Spec_Dist_sum'] = measure['Spec_Dist'] + measure['Dist_Spec']
#         measure['Spec_L1max'] = S1[0] * L1Dist2
#         measure['L1max_Spec'] = S2[0] * L1Dist1
#         measure['Spec_L1max_sum'] = measure['Spec_L1max'] + \
#             measure['L1max_Spec']
#         measure['Dist_Fro'] = Dist1 * Fro2
#         measure['init_Fro'] = S01[0] * Fro2
#         measure['Dist_Fro_sum'] = measure['Dist_Fro'] + measure['init_Fro']

#         # delta is the probability that the generalization bound does not hold
#         delta = 0.01
#         # m is the number of training samples
#         m = len(train_loader.dataset)
#         layer_norm, data_L2, data_Linf, domain_L2 = 0, 0, 0, 0
#         for i, (data, target) in enumerate(train_loader):
#             data = data.to(device).view(target.size(0), -1)
#             layer_out = torch.zeros(target.size(0), H).to(device)

#             # calculate the norm of the output of the first layer in the initial model
#             def fun(m, i, o): layer_out.copy_(o.data)
#             h = init_modules[1].register_forward_hook(fun)
#             output = init_model(data)
#             layer_norm += layer_out.norm(p=2, dim=0) ** 2
#             h.remove()

#             # L2 norm squared of the data data
#             data_L2 += data.norm() ** 2
#             # maximum L2 norm squared on the training set. We use this as an approximation of this quantity over the domain
#             domain_L2 = max(domain_L2, data.norm(p=2, dim=1).max() ** 2)
#             # L_infty norm squared of the data
#             data_Linf += data.max(dim=1)[0].max() ** 2

#         # computing the average
#         data_L2 /= m
#         data_Linf /= m
#         layer_norm /= m

#         # number of parameters
#         measure['#parameter'] = num_param

#         # Generalization bound based on the VC dimension by Harvey et al. 2017
#         VC = (2 + num_param * math.log(8 * math.e * (H + 2 * C) * math.log(4 * math.e * (H + 2 * C), 2), 2)
#               * (2 * (D + 1) * H + (H + 1) * C) / ((D + 1) * H + (H + 1) * C))
#         measure['VC bound'] = 8 * \
#             (C * VC * math.log(math.e * max(m / VC, 1))) + 8 * math.log(2 / delta)

#         # Generalization bound by Bartlett and Mendelson 2002
#         R = 8 * C * L1max1 * L1max2 * 2 * \
#             math.sqrt(math.log(D)) * math.sqrt(data_Linf) / margin
#         measure['L1max bound'] = (R + 3 * math.sqrt(math.log(m / delta))) ** 2

#         # Generalization bound by Neyshabur et al. 2015
#         R = 8 * math.sqrt(C) * Fro1 * Fro2 * math.sqrt(data_L2) / margin
#         measure['Fro bound'] = (R + 3 * math.sqrt(math.log(m / delta))) ** 2

#         # Generalization bound by Bartlett et al. 2017
#         R = (144 * math.log(m) * math.log(2 * num_param) * (math.sqrt(data_L2) + 1 / math.sqrt(m))
#              * (((S2[0] * L1Dist1) ** (2 / 3) + (S1[0] * L1Dist2) ** (2 / 3)) ** (3 / 2)) / margin)
#         measure['Spec_L1 bound'] = (R + math.sqrt(4.5 * math.log(1 / delta) + math.log(2 * m / max(margin, 1e-16))
#                                                   + 2 * math.log(2 + math.sqrt(m * data_L2)) + 2 * math.log((2 + 2 * Dist1)
#                                                                                                             * (2 + 2 * Dist2) * (2 + 2 * S1[0]) * (2 + 2 * S2[0])))) ** 2

#         # Generalization bound by Neyshabur et al. 2018
#         R = (42 * 8 * S1[0] * math.sqrt(math.log(8 * H)) * math.sqrt(domain_L2)
#              * math.sqrt(H * (S2[0] * Dist1) ** 2 + C * (S1[0] * Dist2) ** 2) / (math.sqrt(2) * margin))
#         measure['Spec_Fro bound'] = R ** 2 + 6 * math.log(2 * m / delta)

#         # Our generalization bound
#         R = (3 * math.sqrt(2) * (math.sqrt(2 * C) + 1) * (Fro2 + 1)
#              * (math.sqrt(layer_norm.sum()) + (Dist1 * math.sqrt(data_L2)) + 1) / margin)
#         measure['Our bound'] = (
#             R + 3 * math.sqrt((5 * H + math.log(max(1, margin * math.sqrt(m)) / delta)))) ** 2

#     return measure
import torch
import math
import copy
import warnings
import torch.nn as nn
import pdb


# This function reparametrizes the networks with batch normalization in a way that it calculates the same function as the
# original network but without batch normalization. Instead of removing batch norm completely, we set the bias and mean
# to zero, and scaling and variance to one
# Warning: This function only works for convolutional and fully connected networks. It also assumes that
# module.children() returns the children of a module in the forward pass order. Recurssive construction is allowed.
def reparam(model, prev_layer=None):
    for child in model.children():
        module_name = child._get_name()
        prev_layer = reparam(child, prev_layer)
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            prev_layer = child
        elif module_name in ['BatchNorm2d', 'BatchNorm1d']:
            with torch.no_grad():
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_(
                    child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_(
                    (prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
    return prev_layer


# This function calculates a measure on the given model
# measure_func is a function that returns a value for a given linear or convolutional layer
# calc_measure calculates the values on individual layers and then calculate the final value based on the given operator
def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    measure_val = 0
    if operator == 'product':
        measure_val = math.exp(calc_measure(
            model, init_model, measure_func, 'log_product', kwargs, p))
    elif operator == 'norm':
        measure_val = (calc_measure(model, init_model,
                                    measure_func, 'sum', kwargs, p=p)) ** (1 / p)
    else:
        measure_val = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
                if operator == 'log_product':
                    measure_val += math.log(measure_func(child,
                                                         init_child, **kwargs))
                elif operator == 'sum':
                    measure_val += (measure_func(child,
                                                 init_child, **kwargs)) ** p
                elif operator == 'max':
                    measure_val = max(measure_val, measure_func(
                        child, init_child, **kwargs))
            else:
                measure_val += calc_measure(child, init_child,
                                            measure_func, operator, kwargs, p=p)
    return measure_val

# calculates l_pq norm of the parameter matrix of a layer:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together


def norm(module, init_module, p=2, q=2):
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()

# calculates l_p norm of eigen values of a layer
# convolutional tensors are reshaped in a way that all dimensions except the output are together


def op_norm(module, init_module, p=float('Inf')):
    _, S, _ = module.weight.view(module.weight.size(0), -1).svd()
    return S.norm(p).item()

# calculates l_pq distance of the parameter matrix of a layer from the random initialization:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together


def dist(module, init_module, p=2, q=2):
    return (module.weight - init_module.weight).view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()

# calculates l_pq distance of the parameter matrix of a layer from the random initialization with an extra factor that
# depends on the number of hidden units


def h_dist(module, init_module, p=2, q=2):
    return (n_hidden(module, init_module) ** (1 - 1 / q)) * dist(module, init_module, p=p, q=q)

# ratio of the h_dist to the operator norm


def h_dist_op_norm(module, init_module, p=2, q=2, p_op=float('Inf')):
    return h_dist(module, init_module, p=p, q=q) / op_norm(module, init_module, p=p_op)

# number of hidden units


def n_hidden(module, init_module):
    return module.weight.size(0)

# depth --> always 1 for any linear of convolutional layer


def depth(module, init_module):
    return 1

# number of parameters


def n_param(module, init_module):
    bparam = 0 if module.bias is None else module.bias.size(0)
    return bparam + module.weight.size(0) * module.weight.view(module.weight.size(0), -1).size(1)

# This function calculates path-norm introduced in Neyshabur et al. 2015


def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p)).item()


# This function calculates various measures on the given model and returns two dictionaries:
# 1) measures: different norm based measures on the model
# 2) bounds: different generalization bounds on the model
def calculate(trained_model, init_model, device, train_loader, margin, dataset_name):
    # nchannels, nclasses, img_dim

    nchannels, nclasses, img_dim,  = 3, 10, 32
    if dataset_name.upper() == 'MNIST':
        nchannels = 1
    if dataset_name.upper() == 'CIFAR100':
        nclasses = 100

    model = copy.deepcopy(trained_model)
    reparam(model)
    reparam(init_model)

    # size of the training set
    m = len(train_loader.dataset)

    # depth
    d = calc_measure(model, init_model, depth, 'sum', {})

    # number of parameters (not including batch norm)
    nparam = calc_measure(model, init_model, n_param, 'sum', {})

    measure, bound = {}, {}
    with torch.no_grad():
        measure['L_{1,inf} norm'] = calc_measure(model, init_model, norm, 'product', {
                                                 'p': 1, 'q': float('Inf')}) / margin
        measure['Frobenious norm'] = calc_measure(
            model, init_model, norm, 'product', {'p': 2, 'q': 2}) / margin
        measure['L_{3,1.5} norm'] = calc_measure(
            model, init_model, norm, 'product', {'p': 3, 'q': 1.5}) / margin
        measure['Spectral norm'] = calc_measure(
            model, init_model, op_norm, 'product', {'p': float('Inf')}) / margin
        measure['L_1.5 operator norm'] = calc_measure(
            model, init_model, op_norm, 'product', {'p': 1.5}) / margin
        measure['Trace norm'] = calc_measure(
            model, init_model, op_norm, 'product', {'p': 1}) / margin
        measure['L1_path norm'] = lp_path_norm(model, device, p=1, input_size=[
                                               1, nchannels, img_dim, img_dim]) / margin
        measure['L1.5_path norm'] = lp_path_norm(model, device, p=1.5, input_size=[
                                                 1, nchannels, img_dim, img_dim]) / margin
        measure['L2_path norm'] = lp_path_norm(model, device, p=2, input_size=[
                                               1, nchannels, img_dim, img_dim]) / margin

        # Generalization bounds: constants and additive logarithmic factors are not included

        # This value of alpha is based on the improved depth dependency by Golowith et al. 2018
        alpha = math.sqrt(d + math.log(nchannels * img_dim * img_dim))

        bound['L1_max Bound (Bartlett and Mendelson 2002)'] = alpha * \
            measure['L_{1,inf} norm'] / math.sqrt(m)
        bound['Frobenious Bound (Neyshabur et al. 2015)'] = alpha * \
            measure['Frobenious norm'] / math.sqrt(m)
        bound['L_{3,1.5} Bound (Neyshabur et al. 2015)'] = alpha * \
            measure['L_{3,1.5} norm'] / (m ** (1/3))

        beta = math.log(m) * math.log(nparam)
        ratio = calc_measure(model, init_model, h_dist_op_norm, 'norm', {
                             'p': 2, 'q': 1, 'p_op': float('Inf')}, p=2/3)
        bound['Spec_L_{2,1} Bound (Bartlett et al. 2017)'] = beta * \
            measure['Spectral norm'] * ratio / math.sqrt(m)

        ratio = calc_measure(model, init_model, h_dist_op_norm, 'norm', {
                             'p': 2, 'q': 2, 'p_op': float('Inf')}, p=2)
        bound['Spec_Fro Bound (Neyshabur et al. 2018)'] = d * \
            measure['Spectral norm'] * ratio / math.sqrt(m)

    return measure, bound

    

def store_measure(writer, measure, bound, id_epoch): 
    writer.add_scalar('L_{1,inf} norm', measure['L_{1,inf} norm'], id_epoch+1)
    writer.add_scalar('Frobenious norm', measure['Frobenious norm'], id_epoch+1)
    writer.add_scalar('L_{3,1.5} norm', measure['L_{3,1.5} norm'], id_epoch+1)
    writer.add_scalar('Spectral norm', measure['Spectral norm'], id_epoch+1)
    writer.add_scalar('L_1.5 operator norm', measure['L_1.5 operator norm'], id_epoch+1)
    writer.add_scalar('Trace norm', measure['Trace norm'], id_epoch+1)
    writer.add_scalar('L1_path norm', measure['L1_path norm'], id_epoch+1)
    writer.add_scalar('L1.5_path norm', measure['L1.5_path norm'], id_epoch+1)
    writer.add_scalar('L2_path norm', measure['L2_path norm'], id_epoch+1)
    writer.add_scalar('L1_max Bound (Bartlett and Mendelson 2002)', bound['L1_max Bound (Bartlett and Mendelson 2002)'], id_epoch+1)
    writer.add_scalar('Frobenious Bound (Neyshabur et al. 2015)', bound['Frobenious Bound (Neyshabur et al. 2015)'], id_epoch+1)
    writer.add_scalar('L_{3,1.5} Bound (Neyshabur et al. 2015)', bound['L_{3,1.5} Bound (Neyshabur et al. 2015)'], id_epoch+1)
    writer.add_scalar('Spec_L_{2,1} Bound (Bartlett et al. 2017)', bound['Spec_L_{2,1} Bound (Bartlett et al. 2017)'], id_epoch+1)
    writer.add_scalar('Spec_Fro Bound (Neyshabur et al. 2018)', bound['Spec_Fro Bound (Neyshabur et al. 2018)'], id_epoch+1)

