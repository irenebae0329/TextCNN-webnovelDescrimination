from calendar import EPOCH
import time
import preprocess
from torchtext import vocab
import torch
import torch.nn as nn
import os
from gensim.models import word2vec
import json
import TCNN
resources_path='resources'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=word2vec.Word2Vec.load(os.path.abspath('.')+"/resources/word2vec.model")
vocab=model.wv.key_to_index
label2idx=json.load(open(os.path.join(resources_path,'categories.json')))
text_pipeline=lambda x:[vocab[word] for word in x]
label_pipeline=lambda x:int(label2idx[x])
mydataset=preprocess.Scratch_DataSet(cate_nam="categories.txt",words_nam="words_sentence.txt",vocab=vocab,text_pipeline=text_pipeline,label_pipeline=label_pipeline,device=device,label_nums=len(label2idx))
train_iter,test_iter=mydataset.get_data_iter()
'''
单步迭代过程：
loss = nn.CrossEntropyLoss(reduction='none')
net=TextCNN(config)
trainer = torch.optim.Adam(net.parameters(), lr=0.001)
trainer.zero_grad()
y,x=preprocess.test()
l=loss(net(x),y)
l.sum().backward()
trainer.step()
'''

#超参数
loss = nn.CrossEntropyLoss(reduction='none')    #criterion
net=TCNN.TextCNN(TCNN.config)   #model
trainer = torch.optim.Adam(net.parameters(), lr=0.001)  #optimizer
Epochs=3    #epoch

# train_dataloader=mydataset.get_DataLoader(train_iter)
# test_dataloader=mydataset.get_DataLoader(test_iter)
# print(type(train_dataloader))
# print(type(train_iter))
# print(len(train_dataloader))
# print(len(train_iter))

"""
def train(dataloader,epoch):
    net.train()
    log_interval=300
    start_time=time()
    for idx,(label, text) in enumerate(dataloader):
        trainer.zerograd()
        y,x=preprocess.test()
        l=loss(net(x),y)
        l.sum().backward()
        trainer.step()
        if idx%log_interval==0 and idx>0:
            time_interval=time()-start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | {:5.f}s |').format(epoch,idx,len(dataloader,time_interval))
            start_time = time.time()

for epoch in range(1, Epochs + 1):
    epoch_start_time = time()
    train(train_dataloader,epoch)
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '.format(epoch,time() - epoch_start_time))
    print('-' * 59)
"""

