from gensim.models import word2vec
import io_utils
import argparse
vocab=word2vec.Word2Vec.load('word2vec.model') 
def predict_category(words_list):
    
if __name__=='__main__':
    parser=argparse.ArgumentParser(description="text classfication of novels")
    parser.add_argument('--dir')
    parser.add_argument('--txt')
    parser.add_argument('--stopwords',default="/Users/lwd011204/书籍爬虫/book_spyder/word2vec_model/stop_words.txt")
    args=parser.parse_args()
    if args.dir:
        res=io_utils.read_from_dir(args.dir,stop_words_path=args.stopwords)
    elif args.txt:
        res=io_utils.read_from_txt(args.txt,stop_words_path=args.stopwords)
    print(res)