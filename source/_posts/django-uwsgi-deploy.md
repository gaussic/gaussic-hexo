---
title: Django + uWSGI部署
date: 2016-08-03 00:19:53
tags: [Python, Django, uWSGI]
categories: Technology
---

英文原文请参照此文：[Setting up Django and your web server with uWSGI and nginx](http://uwsgi-docs.readthedocs.org/en/latest/tutorials/Django_and_nginx.html) 我只是稍微翻译并总结了一下。

对于Django部署来说，选择nginx和uWSGI是一个不错的选择，此教程旨在将Django部署到生产环境的服务器中。当然你也可以使用Apache或者其他的服务器部署方式，不过笔者看来，用uWSGI还是相对简单的。

# 概念

Web Server是面向外界的。它可以提供文件服务，但并不能直接与Django应用通话；它需要一些东西来运行这个应用，将请求从客户端喂给它，并且返回响应。

Web Server Gateway Interface - WSGI - 就是用来做这件事的。WSGI是一种Python标准。

uWSGI是WSGI的一种实现。在此教程中，我们将创建uWSGI，以让它创建一个Unix socket，并且通过 WSGI协议来服务于web server的响应。整个最后形式如下：

```
the web client <-> the web server <-> the socket <-> uwsgi <-> Django  
```

# 安装uWSGI之前

## virtualenv

先确保拥有一个python虚拟环境：

```
virtualenv uwsgi-tutorial  
cd uwsgi-tutorial  
source bin/activate
```

<!-- more -->

## 安装Django

将Django安装到你的虚拟环境中，创建一个新的project，并 `cd` 到这个目录下:

```
pip install Django  
django-admin.py startproject mysite  
cd mysite
```

## 关于域和端口

在这篇教程中，我们将称你的域为 example.com，可以自行替换为你的IP。

通篇我们将使用8000端口来部署web服务，就如Django运行环境默认的一样。当然你也可以换成另外的端口，但注意不要与其他应用冲突。

# 基本的uWSGI安装和配置

## 在virtualenv中安装uWSGI

```
pip install uwsgi  
```

注意在安装uwsgi之前请确保安装了python开发包，使用Debian系统的话，安装`pythonX.Y-dev`，X.Y是Python的版本。

## 简单测试

创建一个文件`test.py`:

```
# test.py
def application(env, start_response):  
    start_response('200 OK', [('Content-Type','text/html')])
    return [b"Hello World"] # python3
    #return ["Hello World"] # python2
```

运行uWSGI:

```
uwsgi --http :8000 --wsgi-file test.py  
```

参数意义：

- http :8000：使用http协议，8000端口
- wsgi-file test.py：载入特定文件, test.py

这应该直接在浏览器中返回`hello world`。访问：

```
http://example.com:8000
```

以检查。如果如此，说明如下配置成功了：

```
the web client <-> uWSGI <-> Python  
```

## 测试你的Django project

现在我们想让uWSGI做同样的是，但是是运行一个Django项目，而不是test.py模块。

如果你还未这样做过，请确保你的 `mysite` 项目运行正确：

```
python manage.py runserver 0.0.0.0:8000  
```

如果它成功了，使用uWSGI运行它：

```
uwsgi --http :8000 --module mysite.wsgi  
```

- `module mysite.wsgi`：载入特定wsgi模块
在浏览器中访问你的服务器，如果出现了网站，说明uWSGI可以服务一个Django应用，在virtualenv中，如下：

```
the web client <-> uWSGI <-> Django  
```

现在一般我们不会让浏览器直接与uWSGI对话。这是web server的工作。

# 基本的 nginx

## 安装 Nginx

```
sudo apt-get install nginx  
sudo /etc/init.d/nginx start    # start nginx  
```

安装完后检查nginx正在服务，访问80端口，你应该能得到一个“Welcome to nginx!”的返回。说明：

```
the web client <-> the web server  
```

## 为你的网站配置Nginx

你需要`uwsgi_params`文件，访问[GitHub](https://github.com/nginx/nginx/blob/master/conf/uwsgi_params)下载。

复制到你的项目目录。之后我们会通知Nginx来引用它。

现在，创建一个文件叫做`mysite_nginx.conf`，然后把这些放进去(可以复制default修改)：

```
# mysite_nginx.conf

# the upstream component nginx needs to connect to
upstream django {  
    # server unix:///path/to/your/mysite/mysite.sock; # for a file socket
    server 127.0.0.1:8001; # for a web port socket (we'll use this first)
}

# configuration of the server
server {  
    # the port your site will be served on
    listen      8000;
    # the domain name it will serve for
    server_name .example.com; # substitute your machine's IP address or FQDN
    charset     utf-8;

    # max upload size
    client_max_body_size 75M;   # adjust to taste

    # Django media
    location /media  {
        alias /path/to/your/mysite/media;  # your Django project's media files - amend as required
    }

    location /static {
        alias /path/to/your/mysite/static; # your Django project's static files - amend as required
    }

    # Finally, send all non-media requests to the Django server.
    location / {
        uwsgi_pass  django;
        include     /path/to/your/mysite/uwsgi_params; # the uwsgi_params file you installed
    }
}
```

这一配置文件告诉nginx从文件系统为文件提供服务，以及处理需要Django的请求。

创建一个链接以让nginx发现它：

```
sudo ln -s ~/path/to/your/mysite/mysite_nginx.conf /etc/nginx/sites-enabled/  
```

## 部署静态文件

在运行nginx之前，要把Django的静态文件集中到static文件夹中。首先你应该修改 mysite/settings.py文件，添加：

```
STATIC_ROOT = os.path.join(BASE_DIR, "static/")  
```

然后执行：

```
python manage.py collectstatic  
```

## 基本nginx测试

重启nginx

```
sudo /etc/init.d/nginx restart  
```

为确定Media文件被正确服务，添加一个图片文件`media.png`到`/path/to/your/project/project/media directory`目录，然后访问 [http://example.com:8000/media/media.png](http://example.com:8000/media/media.png)。 如果成功，你将会知道至少nginx服务文件是正常的。

# nginx和uWSGI和test.py

让我们让Nginx来与“hello world” test.py进行通话。

```
uwsgi --socket :8001 --wsgi-file test.py  
```

这几乎与之前的一样，除了参数不同

- `socket :8001`：使用uwsgi协议，8001端口
Nginx同时配置完成了，以与uWSGI通信在8001端口通信，并在外部8000端口通信。访问:

[http://example.com:8000/](http://example.com:8000/)

以检查。现在整个stack如下：

```
the web client <-> the web server <-> the socket <-> uWSGI <-> Python  
```

# 使用Unix sockets代替端口

使用TCP端口socket虽然简单，但是最好使用 Unix sockets而不是端口。

编辑 `mysite_nginx.conf`：

```
server unix:///path/to/your/mysite/mysite.sock; # for a file socket  
# server 127.0.0.1:8001; # for a web port socket (we'll use this first)
```

重启nginx。

重新运行 uWSGI:

```
uwsgi --socket mysite.sock --wsgi-file test.py  
```

Try [http://example.com:8000/](http://example.com:8000/) in the browser.

## 如果不起作用

检查nginx 错误日志 (/var/log/nginx/error.log)。如果你看到如下：

```
connect() to unix:///path/to/your/mysite/mysite.sock failed (13: Permission  
denied)
```

说明权限不够，尝试：

```
uwsgi --socket mysite.sock --wsgi-file test.py --chmod-socket=666 # (very permissive)  
```

或者：

```
uwsgi --socket mysite.sock --wsgi-file test.py --chmod-socket=664 # (more sensible)  
```

# 使用uWSGI和Nginx运行Django项目

现在运行Django项目:

```
uwsgi --socket mysite.sock --module mysite.wsgi --chmod-socket=664  
```

现在uwsgi和nginx应该在服务你的Django应用，而不是hello world。

# 配置uWSGI以使用.ini文件运行

可以将参数放在文件中，然后运行该文件以运行uwsgi。

创建一个文件`mysite_uwsgi.ini`：

```
# mysite_uwsgi.ini file
[uwsgi]

# Django-related settings
# the base directory (full path)
chdir           = /path/to/your/project  
# Django's wsgi file
module          = project.wsgi  
# the virtualenv (full path)
home            = /path/to/virtualenv

# process-related settings
# master
master          = true  
# maximum number of worker processes
processes       = 10  
# the socket (use the full path to be safe
socket          = /path/to/your/project/mysite.sock  
# ... with appropriate permissions - may be needed
# chmod-socket    = 664
# clear environment on exit
vacuum          = true  
```

使用如下命令运行uwsgi：

```
uwsgi --ini mysite_uwsgi.ini # the --ini option is used to specify a file  
```

暂时用到这些，后面的就不翻译了。
