---
title: PIP简易教程
date: 2016-08-02 21:50:46
tags: [Python]
categories: Technology
author: Gaussic
---

安装pip:

```
$ apt-get install python-pip
```

查看pip版本：

```
$ pip --version
pip 8.0.2 from /root/.pyenv/versions/3.5.1/lib/python3.5/site-packages (python 3.5)
```

升级pip：

```
$ pip install -U pip
```

安装包：

```
$ pip install <package>
$ pip install <package>==1.0.4 (特定版本)
$ pip install <package>>=1.0.4 (最小版本)
```

升级已有安装包：

```
$ pip install -U <package>
```

卸载安装包：

```
$ pip uninstall <package>
```

搜索安装包：

```
$ pip search <package>
```

列出已安装的包:

```
$ pip list
$ pip list --outdated (过期安装包)
```

显示已安装包的详细信息：

```
$ pip show <package>
```
