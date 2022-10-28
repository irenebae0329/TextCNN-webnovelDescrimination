from cgi import test
from unicodedata import category
import torchtext
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
class Scratch_DataSet:
    def __init__(self,cate_nam,words_nam,vocab,text_pipeline,label_pipeline,device,label_nums) -> None:
        batches,self.max_length=self.get_scratchData(cate_nam,words_nam)
        self.vocab=vocab
        self.text_pipeline=text_pipeline
        self.label_pipeline=label_pipeline
        self.device=device
        self.label_nums=label_nums
        train,test=self.split_train_test(batches)
        self.test_iter=self.get_DataLoader(test)
        #print(len(train),len(test))
    def get_scratchData(self,cate_num,words_num):
        cate_path=os.path.join(resources_path,cate_num)
        words_path=os.path.join(resources_path,words_num)
        with open(cate_path,'r') as fp:
            categories=fp.read().splitlines()
            fp.close()
        with open(words_path,'r') as fp:
            scratches=fp.read().splitlines()
            fp.close()
        assert len(categories)==len(scratches),"行数不同"
        nums=len(categories)
        batches=[]
        max_length=0
        for i in range(nums):
            scratch=scratches[i].split()
            max_length=max(len(scratch),max_length)
            batch=[categories[i],scratch]
            batches.append(batch)
        return batches,max_length
    def split_train_test(self,samples,ratio=0.1):
        category_size=int(len(samples)/self.label_nums)
        train_set_size=int(len(samples)/self.label_nums*(1-ratio))
        train_set=[]
        test_set=[]
        for i in range(self.label_nums):
            train_set+=samples[i*category_size:i*category_size+train_set_size]
            test_set+=samples[i*category_size+train_set_size:min(len(samples),(i+1)*category_size)]
        return train_set,test_set
    def collate_batch(self,batch):
        label_list,text_list,offsets=[],[],[0]
        for (label,text) in batch:
            processed_text=torch.tensor(self.text_pipeline(text)+(self.max_length-len(self.text_pipeline(text)))*[0])
            label_list.append(self.label_pipeline(label))
            text_list.append(processed_text)
            #offsets.append(processed_text.size(0))
        #offsets=torch.tensor(offsets[:-1]).cumsum(dim=0)
        label_list=torch.tensor(label_list,dtype=torch.int64)
        text_list=torch.cat(text_list).reshape(-1,self.max_length)
        return label_list.to(self.device),text_list.to(self.device)
        #offsets.to(self.device)
    def get_DataLoader(self,data_iter):
        DataLoader=torch.utils.data.DataLoader(data_iter,batch_size=64,shuffle=True,collate_fn=self.collate_batch)
        return DataLoader
class TextClassificationModel(nn.Module):
    def __init__(self,wv):
        super(TextClassificationModel, self).__init__()
        self.wv=wv
        self.embedding = nn.EmbeddingBag(len(self.wv), self.wv.vector_size,sparse=True)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data=torch.tensor(self.wv.vectors)
    def forward(self,text,offset):
        embedded=self.embedding(text,offset)
        return embedded
def test():
    dataset=Scratch_DataSet(cate_nam="categories.txt",words_nam="words_sentence.txt",vocab=vocab,text_pipeline=text_pipeline,label_pipeline=label_pipeline,device=device,label_nums=len(label2idx))
    label,text=next(iter(dataset.test_iter))
    return label,text
