import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# Hàm chuyển đổi số thực sang định dạng fixed-point 16 bit số bù 2
def float_to_fixed_point_16bit_twos_complement(value, int_bits=2, frac_bits=13):
    sign_bit = 1 if value < 0 else 0
    abs_value = abs(value)
    integer_part = int(abs_value)
    integer_bits = bin(integer_part)[2:].zfill(int_bits)
    fractional_part = abs_value - integer_part
    fractional_bits_str = ''
    for _ in range(frac_bits):
        fractional_part *= 2
        bit = int(fractional_part)
        fractional_bits_str += str(bit)
        fractional_part -= bit
    fixed_point_str = integer_bits + fractional_bits_str
    if sign_bit == 1:
        inverted_fixed_point_str = ''.join('1' if bit == '0' else '0' for bit in fixed_point_str)
        fixed_point_value = int(inverted_fixed_point_str, 2) + 1
        fixed_point_str = bin(fixed_point_value)[2:].zfill(int_bits + frac_bits)
    fixed_point_str = str(sign_bit) + fixed_point_str
    fixed_point_value = int(fixed_point_str, 2)
    fixed_point_value = fixed_point_value & 0xFFFF
    return fixed_point_value

# Hàm chuyển đổi fixed-point 16 bit số bù 2 về số thực
def fixed_point_16bit_to_float(fixed_point_value, int_bits=2, frac_bits=13):
    fixed_point_str = f'{fixed_point_value:016b}'
    is_negative = fixed_point_str[0] == '1'
    if is_negative:
        magnitude = int(''.join('1' if bit == '0' else '0' for bit in fixed_point_str[1:]), 2) + 1
        magnitude_str = f'{magnitude:015b}'
    else:
        magnitude_str = fixed_point_str[1:]
    integer_part = int(magnitude_str[0:int_bits], 2)
    fractional_part = 0
    for i, bit in enumerate(magnitude_str[int_bits:]):
        fractional_part += int(bit) * (2 ** -(i + 1))
    float_value = integer_part + fractional_part
    if is_negative:
        float_value = -float_value
    return float_value

# Hàm tiêm lỗi vào trọng số
def inject_faults(H, fault_rate):
    total_weights = H.size
    total_bits = total_weights * 16
    num_faults = int(fault_rate * total_bits)
    
    bits_left = num_faults
    while bits_left > 0:
        b = min(bits_left, random.randint(1, bits_left))
        for _ in range(b):
            idx = random.randint(0, H.size - 1)
            bit_pos = random.randint(0, 15)
            H[idx] ^= 1 << bit_pos
        bits_left -= b
    return H

# Tải mô hình ResNet-18
model = models.resnet18(pretrained=False, num_classes=10)
model.load_state_dict(torch.load('resnet18_cifar10.pth'))

# Lọc ra toàn bộ trọng số (weights)
weights = [param.detach().cpu().numpy() for name, param in model.named_parameters() if 'weight' in name]

# Chuyển toàn bộ trọng số vào mảng 2 chiều H
H = np.concatenate([w.flatten() for w in weights])

# In ra 10 trọng số đầu tiên trước khi chuyển đổi
print("10 trọng số đầu tiên trước khi chuyển đổi:")
print(H[:10])

# Chuyển đổi tất cả các trọng số sang định dạng fixed-point 16 bit số bù 2 và lưu lại vào mảng H
H_fixed = np.array([float_to_fixed_point_16bit_twos_complement(w) for w in H], dtype=np.uint16)

# In ra 10 trọng số đầu tiên sau khi chuyển đổi
print("10 trọng số đầu tiên sau khi chuyển đổi sang fixed-point 16 bit:")
print(H_fixed[:10])

# Tiêm lỗi vào trọng số fixed-point
fault_rate = 0.0001
H_fixed_faulty = inject_faults(H_fixed, fault_rate)

# Chuyển đổi lại các trọng số fixed-point 16 bit trở lại số thực
H_faulty = np.array([fixed_point_16bit_to_float(w) for w in H_fixed_faulty], dtype=np.float32)

# Cập nhật từ mảng H_faulty vào lại các weight trong model, đảm bảo đúng thứ tự ban đầu
weight_shapes = {}
weight_indices = {}  # Lưu trữ chỉ số ban đầu
index = 0
for name, param in model.named_parameters():
    if 'weight' in name:
        weight_shapes[name] = param.data.shape
        flattened_weights = param.data.detach().cpu().numpy().flatten()
        weight_indices[name] = list(range(index, index + len(flattened_weights)))
        index += len(flattened_weights)

for name, param in model.named_parameters():
    if 'weight' in name:
        indices = weight_indices[name]
        param.data = torch.tensor(H_faulty[indices].reshape(weight_shapes[name]), dtype=param.data.dtype).to(param.data.device)

# Đánh giá mô hình sau khi tiêm lỗi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Tải tập dữ liệu test CIFAR-10
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False)

# Hàm đánh giá mô hình
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Đánh giá mô hình
evaluate_model(model, test_loader)

# Lưu trọng số của mô hình sau khi tiêm lỗi vào file .pth
torch.save(model.state_dict(), 'resnet18_cifar10_fault_injected.pth')
