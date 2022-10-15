import pymongo
from bson.objectid import ObjectId
import re
import jieba
def read_from_dataBase():
        client=pymongo.MongoClient(host='localhost',port=27017)
        db=client['nlp_database']
        collections_cate_urls=db['book']
        scratches=[]
        datas=collections_cate_urls.find()
        for data in datas:
            scratches.append(list(data.values())[1])
        return scratches
def preprocess(sentences,stop_words_path):
    def remove(str):
        regrex=r'[^\u4e00-\u9fa5]'
        return re.sub(regrex,'',str)
    def load_stop_words(path):
        return [word.strip() for word in open(path).readlines()]
    def seg_sentence(str):
        seg_Sentence_list=list(jieba.cut(str))
        outstr=''
        for word in seg_Sentence_list:
            if word not in stop_words:
                outstr+=word+" "
        return outstr.strip()
    def save_words_as_txt(processed_words_list):
        with open('words_sentence.txt','w') as f:
            for word in processed_words_list:
                f.write(word+"\n")
            f.close()
    stop_words=load_stop_words(stop_words_path)
    filtered_words_list=[]
    for sentence in sentences:
        for word in seg_sentence(sentence).split(" "):
            filtered_words_list.append(word)
    save_words_as_txt(filtered_words_list)
stop_words_path="/Users/lwd011204/书籍爬虫/book_spyder/module/stop_words.txt"
processed_words_list=preprocess(read_from_dataBase(),stop_words_path=stop_words_path)

    
