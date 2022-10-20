from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train_data=word2vec.LineSentence('/Users/lwd011204/书籍爬虫/book_spyder/words_sentence.txt')

model=word2vec.Word2Vec(train_data,epochs=2,min_count=1)

model.save('word2vec.model')
print(3)