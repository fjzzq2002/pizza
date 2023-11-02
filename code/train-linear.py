import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

d_hidden=256
n_vocab=59
class MyModelD(nn.Module):
    def __init__(self):
        super(MyModelD, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden//2)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = torch.cat([x[:,0],x[:,1]], dim=1)
        x = self.l1(x)
        x = F.relu(x)
        #x = self.l2(x)
        #x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
class MyModelA(nn.Module):
    def __init__(self):
        super(MyModelA, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0]+x[:,1])
        x = F.relu(x)
        #x = self.l2(x)
        #x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
class MyModelX(nn.Module):
    def __init__(self):
        super(MyModelX, self).__init__()
        self.embed1 = nn.Embedding(n_vocab, d_hidden)
        self.embed2 = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed1.weight.data /= (d_hidden//2)**0.5
        self.embed2.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.l1(self.embed1(x[:,0])+self.embed2(x[:,1]))
        x = F.relu(x)
        #x = self.l2(x)
        #x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
class MyModelB(nn.Module):
    def __init__(self):
        super(MyModelB, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.l2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0]+x[:,1])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x
class MyModelC(nn.Module):
    def __init__(self):
        super(MyModelC, self).__init__()
        self.embed = nn.Embedding(n_vocab, d_hidden)
        self.unembed = nn.Embedding(n_vocab, d_hidden)
        self.l1 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.l2 = nn.Linear(d_hidden, d_hidden, bias=True)
        self.embed.weight.data /= (d_hidden//2)**0.5
        self.unembed.weight.data /= d_hidden**0.5
    def backdoor(self, x):
        x = self.embed(x)
        assert len(x.shape)==3 and x.shape[1]==2
        x = self.l1(x[:,0])+self.l1(x[:,1])
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        return x
    def forward(self, x):
        x = self.backdoor(x)
        x = x @ self.unembed.weight.t()
        return x

# make a dataset with pytorch
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
        for i in range(n_vocab):
            for j in range(n_vocab):
                self.data.append([i,j])
    def __getitem__(self, index):
        return torch.tensor(self.data[index],dtype=int),sum(self.data[index])%n_vocab
    def __len__(self):
        return len(self.data)

full_dataset=MyDataset()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=59*59, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=59*59, shuffle=True)

DEVICE='cuda:'+str(random.randint(0,1))
print(DEVICE)
import wandb
import tqdm
def train(config):
    typ=config['model_type']
    models={'A':MyModelA,'B':MyModelB,'C':MyModelC,'D':MyModelD,'X':MyModelX}
    # training loop
    model = models[typ]()
    model.to(DEVICE)
    def norm(model):
        su=0
        for t in model.parameters():
            su+=(t*t).sum().item()
        return su**0.5
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    def cross_entropy_high_precision(logits, labels):
        # Shapes: batch x vocab, batch
        # Cast logits to float64 because log_softmax has a float32 underflow on overly 
        # confident data and can only return multiples of 1.2e-7 (the smallest float x
        # such that 1+x is different from 1 in float32). This leads to loss spikes 
        # and dodgy gradients
        logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
        loss = -torch.mean(prediction_logprobs)
        return loss
    #criterion = nn.CrossEntropyLoss()
    bar = tqdm.tqdm(range(config['epoch']))
    run = wandb.init(reinit=True,config=config,project='modadd_linears')#,settings=wandb.Settings(start_method="spawn"))
    for epoch in bar:
        for i, data in enumerate(train_loader):
            inputs, labels = map(lambda t:t.to(DEVICE),data)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy_high_precision(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss=loss.item()
        aa=('[TRAIN] epoch %d loss: %.3g ' % (epoch + 1, loss.item()))
        # also print validation loss & accuracy
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total = 0
            for i, data in enumerate(test_loader):
                inputs, labels = map(lambda t:t.to(DEVICE),data)
                outputs = model(inputs)
                loss = cross_entropy_high_precision(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            val_loss=total_loss/len(test_loader)
            val_acc=total_correct/total
            cur_norm=norm(model)
            aa+=('[VAL] epoch %d loss: %.3g accuracy: %.3f norm: %.3f' % (epoch + 1, total_loss/len(test_loader), total_correct/total, norm(model)))
        bar.set_description(aa)
        if run:
            run.log({'training_loss': train_loss,
            'validation_loss': val_loss,
            'validation_accuracy': val_acc,
            'parameter_norm': cur_norm})
    return dict(
        model=model,
        config=config,
        dataset = full_dataset,
        run=run
    )

import random
import string

while True:
    letters_and_numbers = string.ascii_lowercase + string.digits.replace('0', '')
    run_name = ''.join(random.choices(letters_and_numbers, k=10))
    print(run_name)
    model_type=random.choice('X')
    C=n_vocab
    config=dict(
        C=n_vocab,
        model_type=model_type,
        n_vocab=n_vocab,
        d_model=d_hidden,
        d_hidden=d_hidden,
        epoch=20000,
        lr=1e-3,
        weight_decay=2.,
        frac=0.8,
        runid=run_name,
    )
    print(config)
    result_modadd=train(config)
    dataset = result_modadd['dataset']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=C*C)
    model = result_modadd['model']
    oo=[[0]*C for _ in range(C)]
    oc=[[0]*C for _ in range(C)]
    for x,y in dataloader:
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        with torch.inference_mode():
            model.eval()
            o=model(x)[:,:]
            o0=o[list(range(len(x))),y]
            o0=o0.cpu()
            x=x.cpu()
            for p,q in zip(o0,x):
                A,B=int(q[0].item()),int(q[1].item())
                oo[(A+B)%C][(A-B)%C]=p.item()
            o[list(range(len(x))),y]=float("-inf")
            o1=o.topk(dim=-1,k=2).values.cpu()
            print(o1)
            for p,q in zip(o1,x):
                A,B=int(q[0].item()),int(q[1].item())
                oc[(A+B)%C][(A-B)%C]=p[0].item()
    # use seaborn to plot the heatmap of oo
    import seaborn as sns
    run=result_modadd['run']
    oo=np.array(oc)
    dd=np.mean(np.std(oo,axis=0))/np.std(oo.flatten())
    print('dd',dd)
    run.summary['distance_dependency']=dd
    run.summary['distance_irrelevancy']=dd
    run.summary['logits']=oo
    mi,mx=np.min(oo),np.max(oo)
    oo=(oo-mi)/(mx-mi)
    run.summary['logits_normalized']=oo
    sns.heatmap(np.array(oo))
    sb=[]
    sx=[]
    sc=[]
    ss=[]
    for i in range(C):
        s=oo[:,i]
        sb.append(np.median(s))
        ss.append(np.mean(s))
        sx.append(np.std(oo[i]))
    print('std(med(col))',np.std(sb))
    print('mean(std(row))',np.mean(sx))
    run.summary['std_med_col']=np.std(sb)
    run.summary['mean_std_row']=np.mean(sx)
    run.summary['std_mean_col']=np.std(ss)
    run.summary['med_std_row']=np.median(sx)
    model_name=f'save/model_{run_name}.pt'
    model=result_modadd['model']
    torch.save(model.state_dict(),model_name)
    import json
    config['func']=None
    with open(f'save/config_{run_name}.json','w') as f:
        json.dump(config,f)
    run.finish()
