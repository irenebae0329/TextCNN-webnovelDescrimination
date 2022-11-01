from os import scandir
from gensim.models import word2vec
import cnn_model.preprocess as preprocess
import io_utils
import argparse
import torch.nn as nn
import torch
from cnn_model import TCNN
model_path="/Users/lwd011204/书籍爬虫/book_spyder/cnn.pt"
vocab=preprocess.vocab
text_pipeline=lambda x:[vocab[word] for word in x if word in list(vocab.keys())]
#list(preprocess.label2idx.keys())[0]
def predict_category(x):
    return list(preprocess.label2idx.keys())[net(x).argmax(1)]
if __name__=='__main__':
    net=TCNN.TextCNN(TCNN.config)
    net.load_state_dict(torch.load("/Users/lwd011204/书籍爬虫/book_spyder/cnn"))
    parser=argparse.ArgumentParser(description="text classfication of novels")
    parser.add_argument('--dir')
    parser.add_argument('--txt')
    parser.add_argument('--stopwords',default="/Users/lwd011204/书籍爬虫/book_spyder/word2vec_model/stop_words.txt")
    args=parser.parse_args()
    predicted_list=[]
    if args.dir:
        res=io_utils.read_from_dir(args.dir,stop_words_path=args.stopwords)
    elif args.txt:
        res=io_utils.read_from_txt(args.txt,stop_words_path=args.stopwords)
    for batch in res:
        precessed_text=text_pipeline(batch[1])
        precessed_text+=[0]*(46-len(precessed_text))
        precessed_text=torch.tensor(precessed_text).reshape(-1,46)
        text=predict_category(precessed_text)
        scratch=batch[0]
        print("简介:{scratch}|\n 预测结果:{text}\n".format(scratch=scratch,text=text))
        print("-------\n")
        #redicted_list.append((predict_category(precessed_text)))
    #print(predicted_list)