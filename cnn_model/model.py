import torch 
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import word2vec
import preprocess
model=preprocess.model
model.vector_size,len(model.wv)
class config:
    num_class=15
    dropout_rate=0.1
    max_text_length=46
    vocab_size=17146
    vec_dim=100
    kernel_sizes=[2,3,4]
    feature_size=100#一维卷积卷积核个数
    weight=torch.tensor(model.wv.vectors)
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
loss = nn.CrossEntropyLoss(reduction='none')
net=TextCNN(config)
trainer = torch.optim.Adam(net.parameters(), lr=0.001)
trainer.zero_grad()
y,x=preprocess.test()
l=loss(net(x),y)
l.sum().backward()
trainer.step()