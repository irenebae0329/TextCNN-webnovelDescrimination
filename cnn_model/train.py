import preprocess
from torchtext import vocab
import torch
import torch.nn as nn
import os
from gensim.models import word2vec
import json
from torch.utils.data import DataLoader
resources_path='resources'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=word2vec.Word2Vec.load("/Users/lwd011204/书籍爬虫/book_spyder/resources/word2vec.model")
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
