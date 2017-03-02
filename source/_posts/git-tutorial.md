---
title: GIT简易教程
date: 2016-08-03 00:07:56
tags: [Git]
categories: Technology
author: Gaussic
---

注：此教程仅供本人记录使用，单做教程可能看不懂。


### 添加用户，以存放项目

```
# useradd -m git
# passwd git
# (输入密码)
# (再次输入密码)
```

![git1](git-tutorial/1.png)

### 创建本地仓库

```
# su git
$ cd ~
$ mkdir test.git
$ cd test.git
$ git init
```

![git2](git-tutorial/2-1.png)

### 添加git用户名和email

```
$ git config --global user.email "git@xxx.com"
$ git config --global user.name "git"
```

![git3](git-tutorial/3.png)

### 添加文件并提交

```
$ echo "hello, git" >> test.txt
$ git add .
$ git commit -m "initial version"
```

![git4](git-tutorial/4.png)

<!-- more -->

### 建立远程仓库

```
$ git remote add origin ssh://git@this.is.your.ip/~/test.git
$ git push origin master
```

![git5](git-tutorial/5-2.png)

### 创建另一台机器的ssh-key

(若之前已创建，本条可忽略)

```
$ ssh-keygen -t rsa -C "dzkang@hotmail.com"
```

这一步生成了idrsa和idrsa.pub两个文件(windows在用户文件夹找)，将idrsa.pub上传到git用户的.ssh目录下，并添加如authorizedkeys文件内

```
$ cat id_rsa.pub >> authorized_keys
```

![git6](git-tutorial/6.png)

### 使用另一台机器克隆项目

(windows下打开git bash)

```
$ git clone ssh://git@xx.xx.xx.xx/~/test.git
```

![git7](git-tutorial/7.png)

可看到git已经将该项目克隆到本地test文件夹

![git8](git-tutorial/8.png)

### 配置一下本机的git用户名和email

```
$ git config --global user.email "dzkang@xxx.com"
$ git config --global user.name "dzkang"
```

### 添加新文件并提交

```
$ echo "just a test" >> test2.txt
$ git add .
$ git commit -m "add test2"
```

![git10](git-tutorial/10.png)

### PUSH到远程仓库

```
$ git push origin master
```

出现报错信息，被远程仓库拒绝：

![git11](git-tutorial/11.png)

修改远程仓库 .git/config配置

```
$ git config receive.denyCurrentBranch ignore
```

![git12](git-tutorial/12-1.png)

这样，每次push成功，在远程服务器输入 git reset --hard能看到最新内容。

引用网上一段话：

![git13](git-tutorial/13.png)
