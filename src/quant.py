from collections import namedtuple
import torch
import torch.nn.functional as F

QTensor = namedtuple("QTensor", ["tensor", "scale", "zero_point"])


def compute_scale_zero_point(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.0
    qmax = 2.0**num_bits - 1.0

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):

    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.0
    qmax = 2.0**num_bits - 1.0

    scale, zero_point = compute_scale_zero_point(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


## Rework Forward pass of Linear and Conv Layers to support Quantisation
def quantize_layer(x, layer, stat, scale_x, zp_x):
    # for both conv and linear layers

    # cache old values
    W = layer.weight.data
    B = layer.bias.data

    # quantise weights, activations are already quantised
    w = quantize_tensor(layer.weight.data)
    b = quantize_tensor(layer.bias.data)

    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()

    # This is Quantisation Artihmetic
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point

    scale_next, zero_point_next = compute_scale_zero_point(
        min_val=stat["min"], max_val=stat["max"]
    )

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
    # n_chans = x.shape[1]
    # running_mu = torch.zeros(n_chans) # zeros are fine for first training iter
    # running_std = torch.ones(n_chans) # ones are fine for first training iter
    # x = F.relu(F.batch_norm(x, running_mu, running_std, training=False, momentum=0.9))

    x = F.max_pool2d(x, 2, 2)

    stats = update_stats(x.clone().view(x.shape[0], -1), stats, "conv2")

    x = model.conv2(x)
    # n_chans = x.shape[1]
    # running_mu = torch.zeros(n_chans) # zeros are fine for first training iter
    # running_std = torch.ones(n_chans) # ones are fine for first training iter
    # x = F.relu(F.batch_norm(x, running_mu, running_std, training=False, momentum=0.9))

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


def forward_quantize(model, x, stats):

    # Quantise before inputting into incoming layers
    x = quantize_tensor(x, min_val=stats["conv1"]["min"], max_val=stats["conv1"]["max"])

    x, scale_next, zero_point_next = quantize_layer(
        x.tensor, model.conv1, stats["conv2"], x.scale, x.zero_point
    )
    x = F.max_pool2d(x, 2, 2)

    x, scale_next, zero_point_next = quantize_layer(
        x, model.conv2, stats["fc"], scale_next, zero_point_next
    )
    x = F.max_pool2d(x, 2, 2)

    x = x.view(-1, 5*5*16)

    x, scale_next, zero_point_next = quantize_layer(
        x, model.fc, stats["fc1"], scale_next, zero_point_next
    )

    x, scale_next, zero_point_next = quantize_layer(
        x, model.fc1, stats["fc2"], scale_next, zero_point_next
    )

    # Back to dequant for final layer
    x = dequantize_tensor(
        QTensor(tensor=x, scale=scale_next, zero_point=zero_point_next)
    )

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
                output = forward_quantize(model, data, stats)
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
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


if __name__ == "__main__":
    import copy
    from torchvision import datasets
    import torchvision.transforms as transforms
    from src.model import LeNet5

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
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1325,), (0.3105,)),
                ]
            ),
        ),
        batch_size=64,
        shuffle=True,
        **kwargs
    )
    stats = gather_stats(q_model, test_loader)
    test_quantize(q_model, test_loader, quant=True, stats=stats)
