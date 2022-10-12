import requests
import pymongo
from pyquery import PyQuery as pq
from parsel import Selector
from collections import defaultdict
client=pymongo.MongoClient(host='localhost',port=27017)
db=client['nlp_database']
collection=db['book']
#headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Safari/605.1.15'}
collection.insert_one({'name':'steven'})