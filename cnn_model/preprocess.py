import torchtext
from torchtext import vocab
import torch
import torch.utils.data as Data
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
def get_scratchData():
    cate_path=os.path.join(resources_path,'categories.txt')
    words_path=os.path.join(resources_path,'words_sentence.txt')
    with open(cate_path,'r') as fp:
        categories=fp.read().splitlines()
        fp.close()
    with open(words_path,'r') as fp:
        scratches=fp.read().splitlines()
        fp.close()
    assert len(categories)==len(scratches),"行数不同"
    nums=len(categories)
    batches=[]
    for i in range(nums):
        batch=[categories[i],scratches[i].split()]
        batches.append(batch)
    return batches
def save_category2id():
    cate_path=os.path.join(resources_path,'categories.txt')
    with open(cate_path,'r') as fp:
        categories=fp.read().splitlines()
        fp.close()
    filterd_list=list(set(categories))
    json_dict={}
    for i,category in enumerate(filterd_list):
        json_dict[category]=i
    with open(os.path.join(resources_path,"categories.json"),'w') as fp:
        json.dump(json_dict,ensure_ascii=False,)
def split_train_test(samples,ratio=0.1):
    border_index=int(-len(samples)*ratio)
    return samples[0:border_index],samples[border_index:-1]
def collate_batch(batch):
    label_list,text_list,offsets=[],[],[0]
    for (label,text) in batch:
        processed_text=torch.tensor(text_pipeline(text))
        label_list.append(label_pipeline(label))
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    offsets=torch.tensor(offsets[:-1]).cumsum(dim=0)
    label_list=torch.tensor(label_list,dtype=torch.int64)
    text_list=torch.cat(text_list)
    return label_list.to(device),text_list.to(device),offsets.to(device)
def get_DataLoader():
    train_iter,text_iter=split_train_test(get_scratchData())
    myDataLoader=DataLoader(train_iter,batch_size=64,shuffle=True,collate_fn=collate_batch)
    return myDataLoader,model