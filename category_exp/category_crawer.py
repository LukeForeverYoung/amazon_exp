
from bs4 import BeautifulSoup

import requests
import re
host='https://www.amazon.com'
root_url='https://www.amazon.com/gp/site-directory'

def dfs(now_link,deep):
    htm=requests.get(now_link,headers={'Connection':'close'})
    htm.close()
    htm=BeautifulSoup(htm.content,'lxml')
    dep=htm.select('#departments')
    if len(dep)==0:
        return None
    dep=dep[0]
    if len(dep.select('a-spacing-micro,s-navigation-indent-1'))!=0:
        child_boxs=dep.select('a-spacing-micro s-navigation-indent-2')
        for c in child_box:
            text=c.a.text
            print('\t'*deep,text)
    input()


res=requests.get(root_url,headers={'Connection':'close'})
res.close()
doc=BeautifulSoup(res.content,'lxml')
c1_boxs=doc.select('.fsdDeptBox')
print(len(c1_boxs))
for i,box in enumerate(c1_boxs):
    if i<15:
        pass
    c1=box.select('.fsdDeptTitle')[0].text
    print(i,c1)
    continue
    child_box=box.select('.a-link-normal,fsdLink,fsdDeptLink')
    print(len(child_box))
    for c2 in child_box:
        print('\t',c2.text)
        if c2['href'][0]=='/':
            next_link=host+c2['href']
        else:
            next_link=c2['href']
        dfs(next_link,2)