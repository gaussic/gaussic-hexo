# gaussic-hexo
hexo blog

hexo

```
npm install hexo-cli -g
hexo init <folder>
npm install
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
git clone https://github.com/Ben02/hexo-theme-Anatole.git themes/anatole
```

编辑 `_config.yml`, 找到 `theme:` 并把那一行改成 `theme: anatole`, 然后增加下面条目:

```
archive_generator:
  per_page: 0
  yearly: false
  monthly: false
  daily: false
```

### Hexo-all-minifier  文件压缩

https://github.com/chenzhutian/hexo-all-minifier

```
npm install hexo-all-minifier --save
```

_config.yml

```
image_minifier:
  enable: true
  interlaced: false
  multipass: false
  optimizationLevel: 2
  pngquant: false
  progressive: false
```

deploy

```
npm install hexo-deployer-git --save
```