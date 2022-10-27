import torch.nn as nn
class TCnnModule(nn.Module):
    def __init__(self,vocab_sizes,vec_dims,label_sizes,embed_weights) -> None:
        super(TCnnModule,self).__init__()
        V=vocab_sizes
        D=vec_dims
        Ci=1
        Co=