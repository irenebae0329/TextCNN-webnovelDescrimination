import requests
import pymongo
from pyquery import PyQuery as pq
from parsel import Selector
from collections import defaultdict
client=pymongo.MongoClient(host='localhost',port=27017)
db=client['nlp_database']
collections_cate_urls=db['book']
headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Safari/605.1.15'}
def scratch_url(url):
    try:
        response=requests.get(url,headers=headers)
        return response.text
    except:
        print('error while scratching'+url)
def get_categories_hrefs():
    main_html=scratch_url('https://www.qidian.com/all/')
    doc=pq(main_html)
    dict={}
    items=doc('body > div.wrap > div.all-pro-wrap.box-center.cf > div.range-sidebar.fl > div.select-list.sl-item.list-act > div.work-filter.type-filter > ul > li > a ').items()
    for node in items:
        if node.text()!='全部':
            dict[node.text()]=node.attr('href')[2:]
    return dict
def save_cates_href_relations(dict):
    dict_1=defaultdict(list)
    def scratch_cate_href(key,url):
        def scratch_and_save_single_page(html):
            selector=Selector(html)
            scratch_list=list(selector.xpath('//body//p[@class="intro"]/text()').getall())
            for scratch in scratch_list:
                print(key,scratch)
                collections_cate_urls.insert_one({key:scratch})
        for i in range(1,6):
            if i==1:
                text=scratch_url(url)
            else:
                cur_url=url+'-page{index}'.format(index=i)
                text=scratch_url(cur_url)
            res=scratch_and_save_single_page(text)
    for key,url in dict.items():
            scratch_cate_href(key,'https://'+url)
            print('error in {url}'.format(url=url))
    return dict_1
def test():
    dict=get_categories_hrefs()
    print(dict)
    scratch_url("https://www.qidian.com/all/chanId21/")
    save_cates_href_relations(dict)
