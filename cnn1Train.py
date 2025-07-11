import torch
import torchvision
import torchvision.transforms as transforms
from models.exp1cnn import lwCNN
from models.exp1Residual import lwResNet6


def test(net, testloader, device):# 评估函数
    net.eval()  # 评估模式
    correct = 0
    total = 0

    with torch.no_grad():  
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)  # 最大概率的类别索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    net.train()  # 回到训练模式
    return acc

def main():
    # 1. 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. 加载 CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64,
        shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64,
        shuffle=False, num_workers=2)

    # 3. 网络、损失函数、优化器
    #net = lwCNN()
    net = lwResNet6()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 4. 训练循环（示例）
    for epoch in range(20):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            #print(inputs.shape)
            outputs = net(inputs)
            #print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                #print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/100:.3f}")
                running_loss = 0.0

        test(net, testloader, device)

    print("Finished Training")


if __name__ == "__main__":
    main()