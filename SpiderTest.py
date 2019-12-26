import urllib.request
import urllib.parse
import re
import os

#添加header，其中Referer必须传，否则会403，User-Agent必须，模仿浏览器访问
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
    'referer':'https://image.baidu.com'
}

url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1577271115837_R&pv=&ic=0&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word={word}"

keyword = input("请输入关键字：")
#转码
keyword = urllib.parse.quote(keyword,'utf-8')

n = 0
j = 1
while(n<300):
    n+=30;#防止拿到重复图片
    url1 = url.format(word=keyword)
    #获取请求
    rep = urllib.request.Request(url1,headers=header)
    #打开网页
    rep = urllib.request.urlopen(rep)
    try:
        #获取网页内容
        html = rep.read().decode('utf-8')
    except:
        print("出错了！")
        error = 1
        print("出错页数："+str(n))
        pass
    #匹配图片正则
    p = re.compile(r"thumbURL.*?\.jpg")
    #h获取正则匹配到的数据，返回list
    s = p.findall(html)

    if os.path.isdir("D://test_pic") != True:
        os.makedirs("D://test_pic")
    #获取图片
    for i in s:
        i = i.replace('thumbURL":"','')
        print(i)
        #保存图片
        urllib.request.urlretrieve(i, r"D://test_pic/pic{num}.jpg".format(num=j))
        j+=1
print("总共图片：" + str(j-1))