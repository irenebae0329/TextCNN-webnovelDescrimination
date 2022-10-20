import os
from word2vec_model.preprocess import preprocess
stop_words_path="/Users/lwd011204/书籍爬虫/book_spyder/word2vec_model/stop_words.txt"
def read_from_dir(path,stop_words_path):
    try:
        flag=True
        lists=os.listdir(path)
    except:
        flag=False
    assert flag,"文件夹不存在"
    res=[]
    for name in lists:
        with open(os.path.join(path,name),'r') as fp:
            for sentence in fp.readlines():
                res.append(preprocess(sentence,stop_words_path).split(" "))
    return res
def read_from_txt(path,stop_words_path):
    res=[]
    try:
        with open(path,'r') as fp:
            flag=True
            for sentence in fp.readlines():
                res.append(preprocess(sentence,stop_words_path))
    except:
        flag=False
    assert flag,"该文件不存在"
    return [res]
#read_from_dir("texts",stop_words_path)

#print(preprocess("薄纱布保暖",stop_words_path))