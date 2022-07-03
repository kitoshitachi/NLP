from torch.nn.functional import cross_entropy, relu
from torch.nn import MaxPool2d, Conv2d, Linear, Module, Dropout
from torch import flatten, no_grad, optim, save, load

class VGGNet(Module):
    def __init__(self, dropout=0.5):
        super(VGGNet, self).__init__()
        self.pool = MaxPool2d(2, 2)
        self.dropout = Dropout(dropout)
        self.conv1 = Conv2d(3, 32, 3, padding=1)
        self.conv2 = Conv2d(32, 32, 3, padding=1)

        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.conv4 = Conv2d(64, 64, 3, padding=1)

        self.conv5 = Conv2d(64, 128, 3, padding=1)
        self.conv6 = Conv2d(128, 128, 3, padding=1)
        self.conv7 = Conv2d(128, 128, 3, padding=1)

        self.fc1 = Linear(4*4*128, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(relu(x))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(relu(x))

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool(relu(x))

        x = flatten(x, 1)
        x = self.dropout(relu(self.fc1(x)))
        x = self.dropout(relu(self.fc2(x)))
        x = relu(self.fc3(x))
        return x


def train(train_loader, epochs=30, lr=1e-3, momentum=0.9, device='cpu', model_name='myVGG', pre_train=False):
    model = VGGNet().to(device)
    if pre_train:
        try:
            model.load_state_dict(load(model_name))
            model.eval()
        except RuntimeError:
            print("model not found!")
            return
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        loss = 0
        if epoch == 6:  # 6 epoch loss increase so reduce lr
            optimizer = optim.SGD(model.parameters(),
                                  lr=lr/10, momentum=momentum)

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            batchloss = cross_entropy(outputs, labels)
            batchloss.backward()

            optimizer.step()
            loss += batchloss.item()
        print(f"epoch: {epoch} loss: {loss / 2000:.3f}")
    save(model.state_dict(), model_name)


def test(test_loader,device='cpu', model_name='myVGG'):
    total = len(test_loader.dataset)
    correct = 0
    model = VGGNet().to(device)
    try:
        model.load_state_dict(load(model_name))
        model.eval()
    except RuntimeError:
        print("model not found!")
        return

    with no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            y = model(images)
            pred_labels = y.max(dim = 1)[1]
            correct += (pred_labels == labels).sum().item()
    print("correct: ", correct)
    print("total: ", total)
    print(f"accuaracy: {correct/float(total)*100}%",)

def test_every_classed(test_loader, classes, device='cpu', model_name='myVGG'):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    model = VGGNet().to(device)
    model.load_state_dict(load(model_name))
    model.eval()
    with no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            y = model(images)
            pred_labels = y.max(dim = 1)[1]
            for label, prediction in zip(labels, pred_labels):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')