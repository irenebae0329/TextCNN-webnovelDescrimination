from calendar import EPOCH
from time import time
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
#超参数
loss = nn.CrossEntropyLoss(reduction='none')    #criterion
net=TCNN.TextCNN(TCNN.config)   #model
trainer = torch.optim.Adam(net.parameters(), lr=0.001)  #optimizer
<<<<<<< HEAD
Epochs=6    #epoch
=======
Epochs=11    #epoch
>>>>>>> ceec9a8d98947a9ab5788fcfa496ad8c43594b85

def train(dataloader,epoch):
    net.train()
    log_interval=5
    start_time=time()
    for idx,(label, text) in enumerate(dataloader):
        trainer.zero_grad()
        y,x=label,text
        l=loss(net(x),y)
        accuracy=evaluate(test_iter)
        l.sum().backward()
        L=l.sum()
        trainer.step()
        if idx%log_interval==0 and idx>0:
            time_interval=time()-start_time
            print(f'| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches | {time_interval:5.2f}s | loss {L:6.2f}')
            start_time = time()
def evaluate(dataloader):
    total_acc,total_count=0,0
    with torch.no_grad():
        for idx,(label,text) in enumerate(dataloader):
            predicted_label=net(text)
            lo=loss(predicted_label,label)
            total_acc+=(predicted_label.argmax(1)==label).sum().item()
            total_count+=label.size(0)
    return total_acc/total_count   
def main():
    for epoch in range(1, Epochs + 1):
        epoch_start_time = time()
        train(train_iter,epoch)
        accuracy=evaluate(test_iter)
        print('-' * 59)
        print(f'| end of epoch {epoch:3d} | time: {time() - epoch_start_time:5.2f}s | accuracy {accuracy:2f}')
        print('-' * 59)
if __name__=='__main__':
    main()