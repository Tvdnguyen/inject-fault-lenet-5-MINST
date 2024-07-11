import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

# Định nghĩa mô hình LeNet-5
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

# Hàm để lật bit
#def flip_bit(value, bit):
#    int_value = np.int8(np.round(value * 127))  # Chuyển số thực sang định dạng 8-bit số nguyên dạng bù 2
#    int_value ^= 1 << bit  # Lật bit
#    return int_value / 127.0  # Chuyển lại thành số thực

    # Hàm lật bit
def flip_bit(value, bit):
    # Chuyển đổi giá trị float thành số nguyên 8 bit và lật bit
    int_value = np.int8(value)
    int_value ^= 1 << bit
    return np.float32(int_value).item()


# Hàm để tiêm lỗi vào trọng số
def inject_faults(model, bit_error_rate):
    weights = [p for p in model.parameters() if p.requires_grad]
    y = sum(p.numel() for p in weights)
    x = int(bit_error_rate * y * 32)  # Tính số bit cần tiêm lỗi theo công thức m = x/(y*32)
    h = 10

    for _ in range(10):
        if h == 0:
            break

        i = random.randint(1, y)
        bit_positions = random.sample(range(y * 8), x)

        weight_index = 0
        for weight in weights:
            numel = weight.numel()
            if i <= numel:
                break
            i -= numel
            weight_index += 1

        weight_data = weights[weight_index].detach().cpu().numpy().flatten()

        for bit_pos in bit_positions:
            weight_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            if weight_idx < len(weight_data):
                weight_data[weight_idx] = flip_bit(weight_data[weight_idx], bit_idx)

        weights[weight_index].data = torch.tensor(weight_data.reshape(weights[weight_index].shape), dtype=weights[weight_index].dtype).to(weights[weight_index].device)
        h -= 1

    return model

# Tải dữ liệu và tiền xử lý
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Khởi tạo mô hình
num_classes = 10
model = LeNet5(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tải trọng số đã lưu
model.load_state_dict(torch.load('lenet5_weights.pth'))
print('Model weights loaded from lenet5_weights.pth')

# Đánh giá mô hình sau khi lượng tử hóa
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network after fault injection: {accuracy}%')
    return accuracy

# Tiến hành tiêm lỗi và đánh giá mô hình
bit_error_rate = 0.0001  # Ví dụ tỉ lệ lỗi bit
results = []

for _ in range(10):
    model_with_faults = LeNet5(num_classes).to(device)
    model_with_faults.load_state_dict(torch.load('lenet5_weights.pth'))
    inject_faults(model_with_faults, bit_error_rate)
    accuracy = evaluate_model(model_with_faults)
    results.append(accuracy)
    print(f'Iteration {_+1}, Bit Error Rate: {bit_error_rate}, Accuracy: {accuracy}%')

# In ra kết quả cuối cùng
average_accuracy = sum(results) / len(results)
print(f'Average Accuracy after fault injection: {average_accuracy}%')
