---
title: 使用Tensorflow训练循环神经网络语言模型
date: 2017-08-24 18:27:59
tags: [TensorFlow, Language Model, RNN]
categories: Deep Learning
---

读了将近一个下午的[TensorFlow Recurrent Neural Network](https://www.tensorflow.org/tutorials/recurrent)教程，翻看其在[PTB](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)上的实现，感觉晦涩难懂，因此参考了部分代码，自己写了一个简化版的Lanugage Model，思路借鉴了Keras的[LSTM text generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)。

## 语言模型介绍

Language Model，即语言模型，其主要思想是，在知道前一部分的词的情况下，推断出下一个最有可能出现的词。例如，知道了 `The fat cat sat on the`，我们认为下一个词为`mat`的可能性比`hat`要大，因为猫更有可能坐在毯子上，而不是帽子上。

这可能被你认为是常识，但是在自然语言处理中，这个任务是可以用概率统计模型来描述的。就拿`The fat cat sat on the mat`来说。我们可能统计出第一个词`The`出现的概率$p(The)$，`The`后面是`fat`的条件概率为$p(fat|The)$，`The fat`同时出现的联合概率：

$$
p(The, fat) = p(The)·p(fat|The)
$$

这个联合概率，就是`The fat`的合理性，即这句话的出现符不符合自然语言的评判标准，通俗点表述就是这是不是句人话。同理，根据链式规则，`The fat cat`的联合概率可求：

$$
p(The, fat, cat) = p(The)·p(fat|The)·p(cat|The, fat)
$$

在知道前面的词为`The cat`的情况下，下一个词为`cat`的概率可以推导出来：

$$
p(cat|The, fat) = \frac{p(The, fat, cat)}{p(The, fat)}
$$

分子是`The fat cat`在语料库中出现的次数，分母是`The fat`在语料库中出现的次数。

因此，`The fat cat sat on the mat`整个句子的合理性同样可以推导，这个句子的合理性即为它的概率。公式化的描述如下：

$$
p(S) = p(w_1, w_2, ···, w_n) =  p(w_1)·p(w_2|w_1)·p(w_3|w_1, w_2)···p(w_n|w_1, w_2, w_3, ···, w_n-1)
$$

>（公式后的n-1应该为下标，插件问题，下同）

那么知道前面一部分
可以看出一个问题，每当计算下一个词的条件概率，需要计算前面所有词的联合概率。这个计算量相当的庞大。并且，一个句子中大部分词同时出现的概率往往少之又少，数据稀疏非常严重，需要一个非常大的语料库来训练。

一个简单的优化是基于马尔科夫假设，下一个词的出现仅与前面的一个或n个词有关。

最简单的情况，下一个词的出现仅仅和前面一个词有关，称之为bigram。

$$
p(S) = p(w_1, w_2, ···, w_n) =  p(w_1)·p(w_2|w_1)·p(w_3|w_2)·p(w_4|w_3)···p(w_n|w_n-1)
$$

再复杂点，下一个词的出现仅和前面两个词有关，称之为trigram。

$$
p(S) = p(w_1, w_2, ···, w_n) =  p(w_1)·p(w_2|w_1)·p(w_3|w_1, w_2)·p(w_4|w_2, w_3)···p(w_n|w_n-2, w_n-1)
$$

这样的条件概率虽然好求，但是会丢失大量的前面的词的信息，有时会对结果产生不良影响。因此如何选择一个有效的n，使得既能简化计算，又能保留大部分的上下文信息。

以上均是传统语言模型的描述。如果不太深究细节，只需要知道，我们的任务是，知道前面n个词，来计算下一个词出现的概率。并且使用语言模型来生成新的文本。

在本文中，我们更加关注的是，如何使用RNN来推测下一个词。

## 数据准备

TensorFlow的官方文档使用的是Mikolov准备好的PTB数据集。我们可以将其下载并解压出来：

```bash
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
```

部分数据如下，不常用的词转换成了`<unk>`标记，数字转换成了N：

```
we 're talking about years ago before anyone heard of asbestos having any questionable properties
there is no asbestos in our products now
neither <unk> nor the researchers who studied the workers were aware of any research on smokers of the kent cigarettes
we have no useful information on whether users are at risk said james a. <unk> of boston 's <unk> cancer institute
the total of N deaths from malignant <unk> lung cancer and <unk> was far higher than expected the researchers said
```

读取文件中的数据，将换行符转换为`<eos>`，然后转换为词的list：

```python
def _read_words(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '<eos>').split()
```

```python
f = _read_words('simple-examples/data/ptb.train.txt')
print(f[:20])
```

得到：

```
['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano', 'guterman', 'hydro-quebec', 'ipo', 'kia', 'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens', 'sim']
```

构建词汇表，词与id互转：

```python
def _build_vocab(filename):
    data = _read_words(filename)

    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id
```

```python
words, words_to_id = _build_vocab('simple-examples/data/ptb.train.txt')
print(words[:10])
print(list(map(lambda x: words_to_id[x], words[:10])))
```

输出：

```
('the', '<unk>', '<eos>', 'N', 'of', 'to', 'a', 'in', 'and', "'s")
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

将一个文件转换为id表示：

```python
def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[x] for x in data if x in word_to_id]
```

```python
words_in_file = _file_to_word_ids('simple-examples/data/ptb.train.txt', words_to_id)
print(words_in_file[:20])
```

词汇表已根据词频进行排序，由于第一句话非英文，所以id靠后。

```
[9980, 9988, 9981, 9989, 9970, 9998, 9971, 9979, 9992, 9997, 9982, 9972, 9993, 9991, 9978, 9983, 9974, 9986, 9999, 9990]
```

将一句话从id列表转换回词：

```python
def to_words(sentence, words):
    return list(map(lambda x: words[x], sentence))
```

将以上函数整合：

```python
def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    words, word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, words, word_to_id
```

以上部分和官方的例子有一定的相似之处。接下来的处理和官方存在很大的不同，主要参考了Keras例程处理文档的操作：

```python
def ptb_producer(raw_data, batch_size=64, num_steps=20, stride=1):
    data_len = len(raw_data)

    sentences = []
    next_words = []
    for i in range(0, data_len - num_steps, stride):
        sentences.append(raw_data[i:(i + num_steps)])
        next_words.append(raw_data[i + num_steps])

    sentences = np.array(sentences)
    next_words = np.array(next_words)

    batch_len = len(sentences) // batch_size
    x = np.reshape(sentences[:(batch_len * batch_size)], \
        [batch_len, batch_size, -1])

    y = np.reshape(next_words[:(batch_len * batch_size)], \
        [batch_len, batch_size])

    return x, y
```

参数解析：

- raw_data: 即`ptb_raw_data()`函数产生的数据
- batch_size: 神经网络使用随机梯度下降，数据按多个批次输出，此为每个批次的数据量
- num_steps: 每个句子的长度，相当于之前描述的n的大小，这在循环神经网络中又称为时序的长度。
- stride: 取数据的步长，决定数据量的大小。

代码解析：

这个函数将一个原始数据list转换为多个批次的数据，即`[batch_len, batch_size, num_steps]`。

首先，程序每一次取了`num_steps`个词作为一个句子，即x，以这`num_steps`个词后面的一个词作为它的下一个预测，即为y。这样，我们首先把原始数据整理成了`batch_len * batch_size`个x和y的表示，类似于已知x求y的分类问题。

为了满足随机梯度下降的需要，我们还需要把数据整理成一个个小的批次，每次喂一个批次的数据给TensorFlow来更新权重，这样，数据就整理为`[batch_len, batch_size, num_steps]`的格式。

打印部分数据：

```python
train_data, valid_data, test_data, words, word_to_id = ptb_raw_data('simple-examples/data')
x_train, y_train = ptb_producer(train_data)
print(x_train.shape)
print(y_train.shape)
```

输出：

```
(14524, 64, 20)
(14524, 64)
```

可见我们得到了14524个批次的数据，每个批次的训练集维度为[64, 20]。

```python
print(' '.join(to_words(x_train[100, 3], words)))
```

第100个批次的第3句话为：

```
despite steady sales growth <eos> magna recently cut its quarterly dividend in half and the company 's class a shares
```

```python
print(words[np.argmax(y_train[100, 3])])
```

它的下一个词为：

```
the
```

## 构建模型
