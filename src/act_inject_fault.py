import torch
import torch.nn.functional as F
import random
import time
import numpy as np
from collections import namedtuple

# Set seed based on current time to ensure different random values each run
random.seed(time.time())

QTensor = namedtuple("QTensor", ["tensor", "scale", "zero_point"])

def compute_scale_zero_point(min_val, max_val, num_bits=8):
    qmin = 0.0
    qmax = 2.0 ** num_bits - 1.0
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = max(qmin, min(initial_zero_point, qmax))
    zero_point = int(zero_point)
    return scale, zero_point



def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        min_val, max_val = x.min(), x.max()

    scale, zero_point = compute_scale_zero_point(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(0, 2 ** num_bits - 1).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def inject_weight_faults(weights, num_faults):
    num_weights = weights.numel()
    weights_flat = weights.flatten().tolist()
    indices = list(range(num_weights))
    while num_faults > 0 and indices:
        if not indices:
            break  
        i = random.choice(indices)
        num_faults_in_weight = random.randint(1, min(8, num_faults))
        weight_bits = weights_flat[i]
        bit_positions = random.sample(range(8), num_faults_in_weight)
        for pos in bit_positions:
            weight_bits ^= (1 << pos)
        weights_flat[i] = weight_bits
        num_faults -= num_faults_in_weight
        indices.remove(i)
    weights = torch.tensor(weights_flat, dtype=torch.uint8).view(weights.shape)
    return weights

def inject_tensor_faults(tensor, num_faults):
    # Kiểm tra xem tensor có phải là QTensor và truy cập tensor thực tế
    if isinstance(tensor, QTensor):
        tensor = tensor.tensor  # Giả sử QTensor có trường 'tensor' là torch.Tensor

    num_tensor = tensor.numel()
    tensor_flat = tensor.flatten().tolist()
    indices = list(range(num_tensor))
    while num_faults > 0 and indices:
        if not indices:
            break
        i = random.choice(indices)
        num_faults_in_tensor = random.randint(1, min(8, num_faults))
        tensor_bits = tensor_flat[i]
        bit_positions = random.sample(range(8), num_faults_in_tensor)
        for pos in bit_positions:
            tensor_bits ^= (1 << pos)
        tensor_flat[i] = tensor_bits
        num_faults -= num_faults_in_tensor
        indices.remove(i)

    # Tạo lại tensor từ danh sách đã thay đổi và trả về định dạng ban đầu
    new_tensor = torch.tensor(tensor_flat, dtype=torch.uint8).view(tensor.shape)
    return new_tensor


def inject_tensor_faults_old(tensor, num_faults):
    num_tensor = tensor.numel()
    tensor_flat = tensor.flatten().tolist()
    indices = list(range(num_tensor))
    while num_faults > 0 and indices:
        if not indices:
            break  
        i = random.choice(indices)
        num_faults_in_tensor = random.randint(1, min(8, num_faults))
        tensor_bits = tensor_flat[i]
        bit_positions = random.sample(range(8), num_faults_in_tensor)
        for pos in bit_positions:
            tensor_bits ^= (1 << pos)
        tensor_flat[i] = tensor_bits
        num_faults -= num_faults_in_tensor
        indices.remove(i)
    tensor = torch.tensor(tensor_flat, dtype=torch.uint8).view(tensor.shape)
    return tensor



## Rework Forward pass of Linear and Conv Layers to support Quantisation
def quantize_layer(x, layer, stat, scale_x, zp_x, num_faults):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights, activations are already quantised
    #w = quantize_tensor_W(layer.weight.data)
    w = quantize_tensor(layer.weight.data)
    b = quantize_tensor(layer.bias.data)

   # Inject faults into weights right after quantization
    w_faulty_tensor = inject_weight_faults(w.tensor, num_faults)  # Get the tensor with faults


    layer.weight.data = w_faulty_tensor.float()
    layer.bias.data = b.tensor.float()

    # This is Quantisation Artihmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = compute_scale_zero_point(min_val=stat["min"], max_val=stat["max"])

    # Preparing input by shifting
    X = x.float() - zp_x
    layer.weight.data = scale_x * scale_w * (layer.weight.data - zp_w)
    layer.bias.data = scale_b * (layer.bias.data + zp_b)

    # All int computation
    x = (layer(X) / scale_next) + zero_point_next

    # Perform relu too
    x = F.relu(x)
    
    # Reset weights for next forward pass
    layer.weight.data = W
    layer.bias.data = B

    
    return x, scale_next, zero_point_next

def quant_feature(x, min_val, max_val, num_faults):
    # Add quantization after activation function
    x = quantize_tensor(x, min_val=min_val, max_val=max_val)
    x = inject_tensor_faults(x, num_faults)
    #x = dequantize_tensor(x)
    return x

## Get Max and Min Stats for Quantising Activations of Network.
# Get Min and max of x tensor, and stores it
def update_stats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]["max"] += max_val.sum().item()
        stats[key]["min"] += min_val.sum().item()
        stats[key]["total"] += 1

    return stats

# Reworked Forward Pass to access activation Stats through update_stats function
def gather_act_stats(model, x, stats):

    stats = update_stats(x.clone().view(x.shape[0], -1), stats, "conv1")
    x = model.conv1(x)
    x = F.max_pool2d(x, 2, 2)
    stats = update_stats(x.clone().view(x.shape[0], -1), stats, "conv2")
    x = model.conv2(x)
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 5*5*16)
    stats = update_stats(x, stats, "fc")
    x = F.relu(model.fc(x))
    stats = update_stats(x, stats, "fc1")
    x = F.relu(model.fc1(x))
    stats = update_stats(x, stats, "fc2")
    x = model.fc2(x)
    return stats

# Entry function to get stats of all functions.
def gather_stats(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gather_act_stats(model, data, stats)
    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {
            "max": value["max"] / value["total"],
            "min": value["min"] / value["total"],
        }
    return final_stats

def calculate_total_faults(fault_rate):
    total_bits = 6508 * 8  # Mỗi trọng số là 8 bits
    total_faults = int(total_bits * fault_rate)
    return total_faults

def distribute_faults_randomly(total_faults, num_parts=4):
    faults = np.random.multinomial(total_faults, np.ones(num_parts) / num_parts)
    return faults.tolist()

def forward_quantize_fix(model, x, stats, fault_rate=0):
    # Tính tổng số bit lỗi từ fault rate
    total_faults = calculate_total_faults(fault_rate)
    # Phân chia ngẫu nhiên tổng số bit lỗi thành bốn phần
    faults_per_layer = distribute_faults_randomly(total_faults, num_parts=4)

    #print(f"Faults per layer: {faults_per_layer}")

    # Lượng tử hóa trước khi đưa vào các lớp tiếp theo
    quantized_x = quantize_tensor(x, min_val=stats["conv1"]["min"], max_val=stats["conv1"]["max"])
    x, scale_next, zero_point_next = quantize_layer(quantized_x.tensor, model.conv1, stats["conv2"], quantized_x.scale, quantized_x.zero_point, 0)
    x = quant_feature(x, min_val=stats["conv1"]["min"], max_val=stats["conv1"]["max"], num_faults= faults_per_layer[0])
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    x = F.max_pool2d(x, 2, 2)
    x, scale_next, zero_point_next = quantize_layer(x, model.conv2, stats["fc"], scale_next, zero_point_next, 0)
    x = quant_feature(x, min_val=stats["conv2"]["min"], max_val=stats["conv2"]["max"], num_faults= faults_per_layer[1])
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 5*5*16)
    x, scale_next, zero_point_next = quantize_layer(x, model.fc, stats["fc1"], scale_next, zero_point_next, 0)
    x = quant_feature(x, min_val=stats["fc"]["min"], max_val=stats["fc"]["max"], num_faults= faults_per_layer[2])
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    x, scale_next, zero_point_next = quantize_layer(x, model.fc1, stats["fc2"], scale_next, zero_point_next, 0)
    x = quant_feature(x, min_val=stats["fc1"]["min"], max_val=stats["fc1"]["max"], num_faults= faults_per_layer[3])
    x = dequantize_tensor(QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next))
    x = model.fc2(x)
    return F.log_softmax(x, dim=1)


def test_quantize(model, test_loader, quant=False, stats=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if quant:
                output = forward_quantize_fix(model, data, stats)
            else:
                output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    #accuracy = 100.0 * correct / len(test_loader.dataset)
    #return accuracy

# Example usage with a simple model (assuming a model and dataset are defined)
if __name__ == "__main__":
    from model import LeNet5
    import copy
    from torchvision import datasets, transforms

    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes).to(device)
    ckpt_path = "./output/lenet5_weights.pth"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    q_model = copy.deepcopy(model)

    kwargs = {"num_workers": 1, "pin_memory": True}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1325,), (0.3105,))
            ]),
            download=True,
        ),
        batch_size=64,
        shuffle=True,
        **kwargs
    )
    stats = gather_stats(q_model, test_loader)
    test_quantize(q_model, test_loader, quant=True, stats=stats)
    # Gather stats and perform test with quantized model and fault injection
    #num_successful_trials = 0
    #total_trials = 0
    #trial_results = []

    #while num_successful_trials < 100:
    #    stats = gather_stats(q_model, test_loader)
    #    accuracy = test_quantize(q_model, test_loader, quant=True, stats=stats)
    #    trial_results.append((total_trials, accuracy))
    #    print(f"Trial {total_trials + 1}: Accuracy = {accuracy:.2f}%")
    #    if accuracy <= 88.902:
    #        trial_results.append(accuracy)
    #        num_successful_trials += 1
    #        print(f"Trial (<=88.902) {num_successful_trials}: Accuracy = {accuracy:.2f}%")
    #    total_trials += 1
    #    # Optional: Save or process trial_results further
    #print("Completed 100 successful trials.")
