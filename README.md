# gaussic-hexo
hexo blog: https://gaussic.github.io


## 安装 node

```
$ brew install node@6
$ brew link node@6
```

(已换成node8)

npm安装插件有时候比较慢，可以使用淘宝的镜像解决：[淘宝 NPM 镜像](https://npm.taobao.org/)

## 安装 hexo

```
$ npm install hexo-cli -g
```

如果安装失败可参照：[npm安装hexo-cli报Building dtrace-provider failed错误](https://github.com/gaussic/code-collector/blob/master/bugs/npm%E5%AE%89%E8%A3%85hexo-cli%E6%8A%A5Building%20dtrace-provider%20failed%E9%94%99%E8%AF%AF.md)

## 设置

```
$ git clone git@github.com:gaussic/gaussic-hexo.git
$ cd gaussic-hexo
$ npm install
```

plugins:

### hexo-asset-image 图片

https://github.com/CodeFalling/hexo-asset-image

```
npm install hexo-asset-image --save
```

_config.yml

```
post_asset_folder: true
```

Example:

```
MacGesture2-Publish
├── apppicker.jpg
├── logo.jpg
└── rules.jpg
MacGesture2-Publish.md
```

### hexo-renderer-jade hexo-generator-archive for anatole

https://munen.cc/tech/Anatole.html

```
npm install --save hexo-renderer-jade hexo-generator-archive
git clone https://github.com/gaussic/hexo-theme-Anatole.git themes/anatole
```

编辑 `_config.yml`, 找到 `theme:` 并把那一行改成 `theme: anatole`, 然后增加下面条目:

```
archive_generator:
  per_page: 0
  yearly: false
  monthly: false
  daily: false
```

deploy

```
npm install hexo-deployer-git --save
```
