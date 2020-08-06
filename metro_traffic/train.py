import copy
import random

import torch
from torch import nn

from metro_traffic.utils import shuffle_jointly


def torch_train_loop(model, data_train, data_test, target_train, target_test, batch_size=32, num_epochs=60,
                     criterion=nn.MSELoss(),
                     optimizer=torch.optim.Adam, weight_decay=0, lr=0.001,
                     valid_check=3, anneal_coeff=0.6, max_not_improved=5, print_every=100000000, cuda=False,
                     seed=12362736):
    random.seed(seed, version=2)
    torch.manual_seed(seed)
    if cuda:
        model = model.cuda()

    optimizer = optimizer(model.parameters(), lr, weight_decay=weight_decay)
    best_loss = float('inf')
    not_improved = 0
    best_model = None
    stride = 48
    step = 5
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        total_loss = 0
        model.train()
        inputs = [0, ] * len(data_train)
        count = 0
        for t in range(0, stride, step):
            data = [0] * len(data_train)
            for i in range(len(data)):
                data[i] = data_train[i][i:i+stride*((data_train[i].shape[0]-i)//stride)]
                data[i] = data[i].reshape(data[i].shape[0]//stride, stride, data[i].shape[-1])
            target = target_train[i:i + stride * ((target_train.shape[0] - i) // stride)]
            target = target.reshape(target.shape[0] // stride, stride, target.shape[-1])
            data = shuffle_jointly(*data)
            for i in range(0, data[0].shape[0], batch_size):
                for j in range(len(data)):
                    inputs[j] = data[j][i:i + batch_size]
                    inputs[j] = torch.from_numpy(inputs[j]).float()

                y = target[i:i + batch_size]
                y = torch.from_numpy(y).float()

                if cuda:
                    y = y.cuda()
                    inputs = [x.cuda() for x in inputs]

                out = model(*inputs)
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1

                if (i // batch_size + 1) % print_every == 0:
                    print('Loss :', total_loss / count)
        print('Training Loss', total_loss / count)
        total_loss = 0
        model.eval()

        count = 0
        for t in range(0, stride, step):
            data = [0] * len(data_train)
            for i in range(len(data)):
                data[i] = data_test[i][i:i+stride*((data_test[i].shape[0]-i)//stride)]
                data[i] = data[i].reshape(data[i].shape[0]//stride, stride, data[i].shape[-1])
            target = target_test[i:i + stride * ((target_test.shape[0] - i) // stride)]
            target = target.reshape(target.shape[0] // stride, stride, target.shape[-1])
            for i in range(0, data[0].shape[0], batch_size):
                for j in range(len(data)):
                    inputs[j] = data[j][i:i + batch_size]
                    inputs[j] = torch.from_numpy(inputs[j]).float()

                y = target[i:i + batch_size]
                y = torch.from_numpy(y).float()

                if cuda:
                    y = y.cuda()
                    inputs = [x.cuda() for x in inputs]

                with torch.no_grad():
                    out = model(*inputs)
                    loss = criterion(out, y)
                total_loss += loss.item()
                count += 1

        total_loss = total_loss / count
        if best_loss > total_loss:
            best_loss = total_loss
            print('New best reached!')
            not_improved = 0
            best_model = copy.deepcopy(model)
        else:
            not_improved += 1

        if (not_improved + 1) % valid_check == 0:
            print('Learning rate decreased... lr=', end='')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * anneal_coeff
                print(param_group['lr'], end=' ')
            print()

        if not_improved > max_not_improved:
            break

        print('Validation Loss', total_loss)
        print('\n')

    print('Best Validation Loss :', best_loss)
    return best_model

def evaluate(model, data_test, target_test, criterion, batch_size, cuda=False):
    stride = 48
    step = 5
    total_loss = torch.zeros(stride)
    if cuda:
        total_loss = total_loss.cuda()
    count = 0
    inputs = [0, ] * len(data_test)
    model.eval()
    for t in range(0, stride, step):
        data = [0] * len(data_test)
        for i in range(len(data)):
            data[i] = data_test[i][i:i + stride * ((data_test[i].shape[0] - i) // stride)]
            data[i] = data[i].reshape(data[i].shape[0] // stride, stride, data[i].shape[-1])
        target = target_test[i:i + stride * ((target_test.shape[0] - i) // stride)]
        target = target.reshape(target.shape[0] // stride, stride, target.shape[-1])
        for i in range(0, data[0].shape[0], batch_size):
            for j in range(len(data)):
                inputs[j] = data[j][i:i + batch_size]
                inputs[j] = torch.from_numpy(inputs[j]).float()

            y = target[i:i + batch_size]
            y = torch.from_numpy(y).float()

            if cuda:
                y = y.cuda()
                inputs = [x.cuda() for x in inputs]
            count += 1
            with torch.no_grad():
                out = model(*inputs)
                loss = criterion(out, y)
            loss = loss.squeeze()
            loss = torch.mean(loss, dim=0)

            total_loss += loss
    return total_loss / count
