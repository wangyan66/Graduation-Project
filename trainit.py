import argparse
import os
import sys
import torch
import torch.nn.functional as F
import data_process
from textcnn import TextCNN
import pandas as pd
import time
import matplotlib.pyplot as plt
from torchsummary import summary

from collections.abc import Iterable

parser = argparse.ArgumentParser(description='TextCNN text classifier')

parser.add_argument('-lr', type=float, default=0.001, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=10)
parser.add_argument('-filter-num', type=int, default=2, help='卷积核的个数')
parser.add_argument('-filter-sizes', type=str, default='2,3', help='不同卷积核大小')
parser.add_argument('-embedding-dim', type=int, default=100, help='词向量的维度')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-static', type=bool, default=True, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1000, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')

args = parser.parse_args()


def get_kfold_data(k, i, X):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    print(X.shape)
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid = X.iloc[val_start:val_end, :]
        X_train = pd.concat((X.iloc[0:val_start, :], X.iloc[val_end:, :]), axis=0)
        # y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:  # 若是最后一折交叉验证
        X_valid = X.iloc[val_start:, :]  # 若不能整除，将多的case放在最后一折里
        X_train = X.iloc[0:val_start, :]
        # y_train = y[0:val_start]

    return X_train, X_valid,


def train(args, train_iter, dev_iter, k):
    # train_iter, dev_iter = data_process.load_data(args) # 将数据分为训练集和验证集
    model = TextCNN(args)
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    last_step = 0
    plt_loss: list = []
    plt_acc: list = []
    plt_TNR: list = []
    plt_TPR: list = []
    plt_FPR: list = []
    plt_PRE: list = []
    plt_F1: list = []
    plt_epoch: list = []
    for epoch in range(1, args.epoch + 1):
        print('*' * 25, 'Epoch:', epoch, '*' * 25)
        stime = time.time()
        Loss_sum, Acc_sum, steps = 0.0, 0.0, 0
        TP, TN, FN, FP = 0, 0, 0, 0
        model.train()
        for batch in train_iter:

            feature, target = batch.text, batch.label
            # t_()函数表示将(max_len, batch_size)转置为(batch_size, max_len)
            with torch.no_grad():
                feature.t_(), target.sub_(1)  # target减去1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)

            # TP    predict 和 label 同时为1
            TP += ((torch.max(logits, 1)[1] == 1) & (target == 1)).cpu().sum()
            # TN    predict 和 label 同时为0
            TN += ((torch.max(logits, 1)[1] == 0) & (target == 0)).cpu().sum()
            # FN    predict 0 label 1
            FN += ((torch.max(logits, 1)[1] == 0) & (target == 1)).cpu().sum()
            # FP    predict 1 label 0
            FP += ((torch.max(logits, 1)[1] == 1) & (target == 0)).cpu().sum()


            loss = F.cross_entropy(logits, target)
            loss.backward()
            Loss_sum += loss.cpu().item()
            corrects = (torch.max(logits, 1)[1] == target).sum()
            Acc_sum += 100.0 * corrects.cpu().item() / batch.batch_size
            optimizer.step()
            steps += 1

            if steps % args.log_interval == 0:
                # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100.0 * corrects.cpu().item() / batch.batch_size

                # Loss = loss.cpu().item()
                # plt_loss.append(Loss)
                # plt_acc.append(train_acc.cpu().item())
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            # if steps % args.test_interval == 0:
            #     # dev_acc = eval(dev_iter, model, args)
            #     dev_acc = test(test_iter,model,args)
            #     if dev_acc > best_acc:
            #         best_acc = dev_acc
            #         last_step = steps
            #         if args.save_best:
            #             print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
            #             save(model, args.save_dir, 'best', steps,epoch)
            # else:
            #     if steps - last_step >= args.early_stopping:
            #         print('\nepoch {} early stop by {} steps, acc: {:.4f}%'.format(epoch,args.early_stopping, best_acc))
            #         # raise KeyboardInterrupt
            #         break;

        plt_TPR.append(TP / (TP + FN))
        plt_TNR.append(TN / (FP + TN))
        plt_FPR.append(FP / (FP + TN))
        Pre = TP / (TP + FP)
        recall = TP / (TP + FN)
        plt_PRE.append(Pre)
        plt_F1.append(2*(Pre*recall)/(Pre+recall))
        plt_loss.append(Loss_sum / steps)
        plt_acc.append(Acc_sum / steps)
        plt_epoch.append(epoch)
        dev_acc = eval(dev_iter, model, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            last_step = steps
            if args.save_best:
                print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                save(model, args.save_dir, 'best', steps, epoch)
        list_tmp: list = [obj for obj in range(0, len(plt_loss))]

        etime = time.time()
        print('第', epoch, '个epoch耗时：{:.4f}s'.format(etime - stime))
    plt.figure(0)
    plt.plot(plt_epoch, plt_loss, label='Loss')
    plt.title("TextCNN")
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.savefig('plot_dir/merge/fold_{}_loss.png'.format(k))

    plt.figure(1)
    plt.plot(plt_epoch, plt_acc, label='Train acc')
    plt.title("TextCNN")
    plt.ylabel('Train acc')
    plt.xlabel('epoch')
    plt.savefig('plot_dir/merge/fold_{}_acc.png'.format(k))

    plt.figure(2)
    plt.plot(plt_epoch, plt_PRE, label='Precision')
    plt.title("TextCNN")
    plt.ylabel('Precision')
    plt.xlabel('epoch')
    plt.savefig('plot_dir/merge/fold_{}_Precision.png'.format(k))

    plt.figure(3)
    plt.plot(plt_epoch, plt_F1, label='F1')
    plt.title("TextCNN")
    plt.ylabel('F1')
    plt.xlabel('epoch')
    plt.savefig('plot_dir/merge/fold_{}_F1.png'.format(k))

    plt.figure(4)
    plt.plot(plt_epoch, plt_FPR, label='FPR')
    plt.title("TextCNN")
    plt.ylabel('FPR')
    plt.xlabel('epoch')
    plt.savefig('plot_dir/merge/fold_{}_FPR.png'.format(k))

    plt.figure(5)
    plt.plot(plt_epoch, plt_TPR, label='TPR')
    plt.title("TextCNN")
    plt.ylabel('TPR')
    plt.xlabel('epoch')
    plt.savefig('plot_dir/merge/fold_{}_TPR.png'.format(k))

    plt.figure(6)
    plt.plot(plt_FPR, plt_TPR, label='AUC')
    plt.title("TextCNN")
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.savefig('plot_dir/merge/fold_{}_AUC.png'.format(k))
    return model


'''
对验证集进行测试 
'''


def eval(data_iter, model, args):
    corrects, avg_loss = 0, 0
    model.eval()
    for batch in data_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature.t_(), target.sub_(1)  # target减去1
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.cpu().item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()) == target).sum().cpu().item()
    size = len(data_iter.dataset)
    avg_loss /= len(data_iter)
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, save_dir, save_prefix, steps, epochs):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_epoch_{}_steps_{}.pt'.format(save_prefix, epochs, steps)
    torch.save(model.state_dict(), save_path)


def k_fold_train(args):
    # train_loss_sum, valid_loss_sum = 0, 0
    test_acc_sum, test_loss_sum = 0, 0
    k = 10
    # path_test = 'data/tmp/postive_samples.csv'
    data_path = 'data_test/data/data.csv'
    test_path = 'data_test/data/data6.csv'
    csv_data = pd.read_csv(data_path, header=0)
    test_data = pd.read_csv(test_path, header=0)
    for i in range(k):
        stime = time.time()
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        train_data, valid_data = get_kfold_data(k, i, csv_data)  # 获取k折交叉验证的训练和验证数据
        train_iter, dev_iter, test_iter = data_process.load_data(args, train_data, valid_data, test_data)  # 返回对应的迭代器
        print('数据处理完成')
        model = train(args, train_iter, dev_iter, i)  # 开始训练
        # 每份数据进行训练
        # train_acc, val_acc = traink(snet, *data, batch_size, learning_rate,  num_epochs) 
        test_acc, test_loss = test(test_iter, model, args)
        etime = time.time()

        print('第', i + 1, '折耗时：{:.4f}s'.format(etime - stime))
        print('test_acc:{:.4f}%'.format(test_acc))
        print('test_loss:{:.4f}\n'.format(test_loss))

        # train_loss_sum += train_loss[-1]
        # valid_loss_sum += val_loss[-1]
        test_acc_sum += test_acc
        test_loss_sum += test_loss

    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)

    print('average test accuracy:{:.4f}%'.format(test_acc_sum / k))
    print('average test loss:{:.4f}'.format(test_loss_sum / k))

    return


def test(data_iter, model, args):
    corrects, avg_loss = 0, 0
    model.eval()
    for batch in data_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature.t_(), target.sub_(1)  # target减去1
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.cpu().item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()) == target).sum().cpu().item()
    size = len(data_iter.dataset)
    avg_loss /= len(data_iter)
    accuracy = 100.0 * corrects / size
    print('\ntest - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                 accuracy,
                                                                 corrects,
                                                                 size))
    return accuracy, avg_loss


k_fold_train(args)
