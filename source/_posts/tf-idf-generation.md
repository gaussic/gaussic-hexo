---
title: TF-IDF关键词提取实现
date: 2017-08-08 17:30:30
tags: [Python, Text, TF-IDF]
categories: Text Processing
---

> 本文旨在对特定的语料库生成各词的逆文档频率。然后根据TF-IDF算法进行关键词提取。

转载请注明出处：[Gaussic](https://gaussic.github.io/) 。

GitHub代码：[https://github.com/gaussic/tf-idf-keyword](https://github.com/gaussic/tf-idf-keyword)

### 分词

对于中文文本的关键词提取，需要先进行分词操作。

去除其中的一些英文和数字，只保留中文：

```python
import jieba
import re

def segment(sentence, cut_all=False):
    sentence = sentence.replace('\n', '').replace('\u3000', '').replace('\u00A0', '')
    sentence = ' '.join(jieba.cut(sentence, cut_all=cut_all))
    return re.sub('[a-zA-Z0-9.。:：,，)）(（！!??”“\"]', '', sentence).split()
```

### 语料库逆文档频率统计

#### 高效文件读取

读取指定目录下的所有文本文件，使用结巴分词器进行分词。本文的IDF提取基于[THUCNews清华新闻语料库](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)的大约80万篇文本。

基于python生成器的实现，以下代码可以实现高效地读取文本并分词：

```python
class MyDocuments(object):    # memory efficient data streaming
    def __init__(self, dirname):
        self.dirname = dirname
        if not os.path.isdir(dirname):
            print(dirname, '- not a directory!')
            sys.exit()

    def __iter__(self):
        for dirfile in os.walk(self.dirname):
            for fname in dirfile[2]:
                text = open(os.path.join(dirfile[0], fname),
                            'r', encoding='utf-8', errors='ignore').read()
                yield segment(text)   # time consuming
```

#### 词的逆文档频数统计

统计每一个词出现在多少篇文档中：

```python
documents = MyDocuments(inputdir)

ignored = {'', ' ', '', '。', '：', '，', '）', '（', '！', '?', '”', '“'}
id_freq = {}
i = 0
for doc in documents:
    doc = set(x for x in doc if x not in ignored)
    for x in doc:
        id_freq[x] = id_freq.get(x, 0) + 1
    if i % 1000 == 0:
        print('Documents processed: ', i, ', time: ',
            datetime.datetime.now())
    i += 1
```

计算逆文档频率并存储

```python
with open(outputfile, 'w', encoding='utf-8') as f:
    for key, value in id_freq.items():
        f.write(key + ' ' + str(math.log(i / value, 2)) + '\n')
```

逆文档频率(IDF)计算公式

$$
IDF(w) = log_2(\frac{D}{D_w})
$$

其中，$D$表示总文档数，$D_w$表示词w出现在多少篇文档中。

#### 运行示例：

```
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/65/1sj9q72d15gg80vt9c70v9d80000gn/T/jieba.cache
Loading model cost 0.943 seconds.
Prefix dict has been built succesfully.
Documents processed:  0 , time:  2017-08-08 17:11:15.906739
Documents processed:  1000 , time:  2017-08-08 17:11:18.857246
Documents processed:  2000 , time:  2017-08-08 17:11:21.762615
Documents processed:  3000 , time:  2017-08-08 17:11:24.534753
Documents processed:  4000 , time:  2017-08-08 17:11:27.235600
Documents processed:  5000 , time:  2017-08-08 17:11:29.974688
Documents processed:  6000 , time:  2017-08-08 17:11:32.818768
Documents processed:  7000 , time:  2017-08-08 17:11:35.797916
Documents processed:  8000 , time:  2017-08-08 17:11:39.232018
```

可见，处理1000篇文档用时大约3秒，80万篇大约用时40分钟。

### TF-IDF关键词提取

借鉴了结巴分词的处理思路，使用IDFLoader载入IDF文件：

```python
class IDFLoader(object):
    def __init__(self, idf_path):
        self.idf_path = idf_path
        self.idf_freq = {}     # idf
        self.mean_idf = 0.0    # 均值
        self.load_idf()

    def load_idf(self):       # 从文件中载入idf
        cnt = 0
        with open(self.idf_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    word, freq = line.strip().split(' ')
                    cnt += 1
                except Exception as e:
                    pass
                self.idf_freq[word] = float(freq)

        print('Vocabularies loaded: %d' % cnt)
        self.mean_idf = sum(self.idf_freq.values()) / cnt
```

使用TF-IDF抽取关键词。TF-IDF计算公式：

$$
TFIDF(w) = TF(w) * IDF(w)
$$

```python
class TFIDF(object):
    def __init__(self, idf_path):
        self.idf_loader = IDFLoader(idf_path)
        self.idf_freq = self.idf_loader.idf_freq
        self.mean_idf = self.idf_loader.mean_idf

    def extract_keywords(self, sentence, topK=20):    # 提取关键词
        # 过滤
        seg_list = segment(sentence)

        freq = {}
        for w in seg_list:
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())

        for k in freq:   # 计算 TF-IDF
            freq[k] *= self.idf_freq.get(k, self.mean_idf) / total

        tags = sorted(freq, key=freq.__getitem__, reverse=True)  # 排序

        if topK:
            return tags[:topK]
        else:
            return tags
```

使用：

```python
# idffile为idf文件路径, document为待处理文本路径
tdidf = TFIDF(idffile)
sentence = open(document, 'r', encoding='utf-8', errors='ignore').read()
tags = tdidf.extract_keywords(sentence, topK)
```

原文档：

```
AMD力推812核服务器处理器反攻英特尔
　　AMD今日正式推出最新的8核心及12核心系列处理器产品，从而正式在服务器领域向英特尔吹起了进攻的号角。
　　AMD的8核和12核服务器处理器都采用了新的45纳米设计，而且也都是由两块处理器die封装在一起构建，其中12核心处理器正是基于此前曝光的Magny-Cours核心，也就是两个6核伊斯坦布尔核心封装在一起，而8核处理器则是由两颗4核处理器die封装在一起构建。
　　新推出的8核和12核处理器将支持全新的G34插槽，可提供更新的I/O技术，另外由于可以支持四条DDR3内存通道因此每颗处理器可以支持多达12条内存插槽。
　　此次新推的8核和12核处理器产品将会隶属于Opteron 6100系列，最低起始主频为1.8GHz，其中8核最低版本型号为Opteron 6124 HE，而该系列最高版本则为主频2.3GHz的12核Opteron 6176 SE。在Opteron 6100系列里，1.8GHz的8核Opteron 6124 HE功耗较低仅为65W，具体的售价则为455美元，折合人民币3100元出头。主频2.3GHz的12核Opteron 6176 SE功耗为105W，售价为1386美元，折合人民币约为9466元。其他产品的规格和价格多介于这两款产品之间。
　　性能方面，AMD Opteron 6100系列比此前的6核伊斯坦布尔处理器要强悍很多，按照AMD方面的说法整数运算性能提升达88%，同时浮点运算性能更是提升了119%之多。Opteron 6000系列服务器平台主要将配备四个或者两个插槽，也就是说入门级系统核心数量为16个，而高阶版系统核心数量可达48个。
　　与AMD相对的是英特尔也正计划针对多处理器服务器市场推出一款8核心的芯片产品，这款产品也被称为“Nehalem-EX”，这款产品应该也已经离正式上市不远。
```

示例输出：

```
核
处理器
服务器
系统核心
封装
系列
插槽
核心
主频
产品
伊斯坦布尔
英特尔
功耗
多处理器
低仅
折合
浮点运算
性能
构建
吹起
```
