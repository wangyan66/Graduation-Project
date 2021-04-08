import argparse
import os
import sys
import torch
import torch.nn.functional as F
import data_process
from textcnn import TextCNN

parser = argparse.ArgumentParser(description='TextCNN text classifier')

parser.add_argument('-lr', type=float, default=0.001, help='学习率')
parser.add_argument('-batch-size', type=int, default=128)
parser.add_argument('-epoch', type=int, default=20)
parser.add_argument('-filter-num', type=int, default=100, help='卷积核的个数')
parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='不同卷积核大小')
parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
parser.add_argument('-dropout', type=float, default=0.5)
parser.add_argument('-label-num', type=int, default=2, help='标签个数')
parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
parser.add_argument('-cuda', type=bool, default=False)
parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
parser.add_argument('-test-interval', type=int, default=100,help='经过多少iteration对验证集进行测试')
parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')

args = parser.parse_args()

def get_kfold_data(k, i, X):
     
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid = X[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim = 0)
        # y_train = torch.cat((y[0:val_start], y[val_end:]), dim = 0)
    else:  # 若是最后一折交叉验证
        X_valid = X[val_start:]    # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        # y_train = y[0:val_start]
        
    return X_train, X_valid,

def train(args,train_iter,dev_iter):
    # print(args.vocab_size)
    # train_iter, dev_iter = data_process.load_data(args) # 将数据分为训练集和验证集
    # print('加载数据完成')
    # print(args.vocab_size)
    model = TextCNN(args)
    if args.cuda: model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epoch + 1):
        print("*********Epoch:%d",epoch)
        for batch in train_iter:
            feature, target = batch.text, batch.label
            # t_()函数表示将(max_len, batch_size)转置为(batch_size, max_len)
            with torch.no_grad():
                feature.t_(),target.sub_(1) # target减去1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                # torch.max(logits, 1)函数：返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                corrects = (torch.max(logits, 1)[1] == target).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                # else:
                #     if steps - last_step >= args.early_stopping:
                #         print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                #         raise KeyboardInterrupt
    return train_acc,dev_acc

'''
对验证集进行测试 
'''
def eval(data_iter, model, args):
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        with torch.no_grad():
            feature.t_(),target.sub_(1) # target减去1
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()) == target).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
def k_fold_train(args){

    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum , valid_acc_sum = 0, 0
    
    for i in range(k):
        print('*'*25,'第', i + 1,'折','*'*25)
        trian_data,valid_data = get_kfold_data(k, i, X_train)    # 获取k折交叉验证的训练和验证数据
        train_iter, dev_iter = data_process.load_data(args,train_data,valid_data) # 返回对应的迭代器
        print('数据处理完成')
        train(args,train_iter,dev_iter)#开始训练
        # 每份数据进行训练
        train_loss, val_loss, train_acc, val_acc = traink(snet, *data, batch_size, learning_rate,  num_epochs) 
       
        
        print('train_acc:{:.3f}%'.format(train_acc[-1]))
        print('valid_acc:{:.3f}%\n'.format(val_acc[-1]))
        
        # train_loss_sum += train_loss[-1]
        # valid_loss_sum += val_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += val_acc[-1]
        
    print('\n', '#'*10,'最终k折交叉验证结果','#'*10) 
    

    print('average train accuracy:{:.3f}%'.format(train_acc_sum/k))
    print('average valid accuracy:{:.3f}%'.format(valid_acc_sum/k))

    return 
}
k_fold_train(args)
