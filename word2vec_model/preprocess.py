import pymongo
import re
import jieba
stop_words_path="/Users/lwd011204/书籍爬虫/book_spyder/word2vec_model/stop_words.txt"
def read_from_dataBase():
    client=pymongo.MongoClient(host='localhost',port=27017)
    db=client['nlp_database']
    collections_cate_urls=db['book']
    scratches=[]
    datas=collections_cate_urls.find()
    for data in datas:
        scratches.append(list(data.values())[1])
    return scratches
def preprocess(sentence,stop_words_path):
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
    stop_words=load_stop_words(stop_words_path)
    return seg_sentence(remove(sentence))

def save_words_as_txt(processed_words_list):
        with open('words_sentence.txt','w') as f:
            for sentence in processed_words_list:
                f.write(sentence+"\n")
            f.close()
def main():
    sentences=read_from_dataBase()
    filtered_words_list=[]
    for sentence in sentences:
        filtered_words_list.append(preprocess(sentence,stop_words_path))
    processed_words_list=preprocess(sentences,stop_words_path=stop_words_path)
    save_words_as_txt(processed_words_list=processed_words_list)
if __name__=='__main__':
    main()