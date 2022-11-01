import torch 
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import word2vec
#mport cnn_model.preprocess as preprocess
model=word2vec.Word2Vec.load("/Users/lwd011204/书籍爬虫/book_spyder/resources/word2vec.model")
print(model.vector_size,len(model.wv))
class config:
    num_class=15#分类个数，可由ScratchDataset中的成员变量得到
    dropout_rate=0.1#人为设定
    max_text_length=46#句子最长长度，ScratchDataset成员变量
    vocab_size=17146#len(model.wv)
    vec_dim=100#词向量维度，model.vector
    kernel_sizes=[2,3,4]#人为设定
    feature_size=100#一维卷积卷积核个数
    weight=torch.tensor(model.wv.vectors)#由预训练模型得到,word2vec.wv.vectors
    vocab_size=len(model.wv)
class TextCNN(nn.Module):
    def __init__(self,config) -> None:
        super(TextCNN,self).__init__()
        self.dropout_rate=config.dropout_rate
        self.num_class=config.num_class
        self.config=config
        self.embedding=nn.Embedding(config.vocab_size,config.vec_dim)
        self.convs=nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=config.vec_dim,out_channels=config.feature_size,kernel_size=h),
        nn.ReLU(),nn.MaxPool1d(kernel_size=config.max_text_length-h+1)) 
        for h in config.kernel_sizes])
        self.fc=nn.Linear(config.feature_size*len(config.kernel_sizes),out_features=config.num_class)
        self.init_weight(config)
    def init_weight(self,config):
        self.embedding.weight.data=config.weight
    def forward(self,x):
        embed_x=self.embedding(x)
        embed_x=embed_x.permute(0,2,1)
        out=[conv(embed_x) for conv in self.convs]
        out=torch.cat(out,dim=1)
        out=out.view(-1,out.size(1))
        out=F.dropout(out,p=self.dropout_rate)
        out=self.fc(out)
        return out
