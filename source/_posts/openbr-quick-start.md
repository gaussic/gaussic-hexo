---
title: OpenBr快速入门
date: 2016-08-05 13:35:12
tags: [CV, Machine Learning, OpenBR]
categories: Computer Vision
author: Gaussic
---

官方翻译加实践，基于Windows版本。

官网地址：[这是链接](http://openbiometrics.org/docs/tutorials/#quick-start)。

这篇教程旨在使用一些有趣的例子让你熟悉OpenBR背后的思想、对象以及动机。注意需要摄像头的支持。

OpenBR是一个基于QT、OpenCV和Eigen而构建的C++库。它既可以在命令行使用`br`命令来使用，还可以通过C++或C的API接口来使用。使用`br`命令是最简单也是最快地起步方法，这篇教程中的所有例子都是基于`br`命令的。

首先，确认OpenBR正确地安装。

Windows版本的安装教程：[这是Windows版教程](https://gaussic.github.io/2016/08/05/openbr-install-build/)。

如果是其他版本，请参照官网：[官网](http://openbiometrics.org/docs/install/)。

> 官方文档存在一定错误，Windows版本可参照上面的链接。

在终端或命令行输入：

```
$ br -gui -algorithm "Show(false)" -enroll 0.webcam
```

注：如果是Windows用户请切换到 `openbr\build-msvc2013\install\bin` 目录下，也可以把这个目录加到环境变量里面。

如果每一步都按照上面进行操作，你的摄像头应该打开了并且开始捕捉视频了。恭喜你，你正在使用OpenBR。

现在我们来聊聊上面的命令到底发生了什么。`-gui`, `-algorithm`和`enroll`是OpenBR的一些flag，它们被用来指定`br`应用的指令操作。OpenBR规定所有的flag都带有`-`前缀，以及所有的参数都用空格隔开。Flags通常需要特定数量的参数。所有可能的flags以及它们的值在这里：[CL_API](http://openbiometrics.org/docs/api_docs/cl_api/)。

让我们一个个解析一些这些参数和值：

- `-gui`是用来告诉OpenBR打开一个GUI窗口的flag。注意，如果使用`-gui`，它必须是第一个传给`br`的flag。
- `-algorithm`是OpenBR最重要的flags之一。它需要一个参数，被称作算法串(algorithm string)。这个字符串决定了传输哪些图像以及元数据的管道。它由[Transforms](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)组成，浙江在后续的教程里讲解。
- `-enroll`从[Gallery](http://openbiometrics.org/docs/api_docs/cpp_api/gallery/gallery/)或[Format](http://openbiometrics.org/docs/api_docs/cpp_api/format/format/)那里读文件，并且加入到算法管道中，以及将它们序列化给另外的[Gallery](http://openbiometrics.org/docs/api_docs/cpp_api/gallery/gallery/)或[Format](http://openbiometrics.org/docs/api_docs/cpp_api/format/format/)。`-enroll`需要一个输入参数(在这个例子中是`0.webcam`)以及一个可选的输出参数。OpenBR支持多种格式，包括`.jpg`, `.png`, `.csv`和`xml`。`.webcam`格式告诉OpenBR从计算机的摄像头采集图像帧作为输入。

<!-- more -->

让我们来试试一个稍微更复杂一点的例子。毕竟，OpenBR能做更多的事情，而不仅仅是开摄像头。再次打开终端输入：

```
$ br -gui -algorithm "Cvt(Gray)+Show(false)" -enroll 0.webcam
```

这里，通过简单地在算法串中添加`Cvt(Gray)`，我们输入普通的BGR(这里是OpenCV的BGR模式)图像并且将其转换成了灰度图像。[Cvt](http://openbiometrics.org/docs/plugin_docs/imgproc/#cvttransform)，是 convert的缩写，是OpenBR [Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)的一个例子，[Show](http://openbiometrics.org/docs/plugin_docs/gui/#showtransform)也是。实际上，OpenBR中的每一个算法串都是组成一个管道的一系列[Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)的结合，甚至连`+`都是[Pipe](http://openbiometrics.org/docs/plugin_docs/core/#pipetransform)的缩写，这是另外一种OpenBR [Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)。

通常，[Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)s会接收参数。我们指定`Gray`作为Cvt的一个运行时参数，来告诉这个[Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)需要将图像转换到哪个颜色空间。我们也可以写`Cvt(HSV)`，如果我们想转换为HSV颜色空间，或者`Cvt(Luv)`，如果我们想转换为LUV。参数可以通过键值对的方式来提供(`Cvt(Gray)`等价于`Cvt(colorSpace=Gray)`)。注意，如果只想传入值的话，请按照算法定义的参数顺序来传值。试试将上面的算法串改为`Show(true)`来看看修改参数对输出的影响(提示：按住一个键然后查看变化)。

我们来把这个例子变得更加刺激以及更加贴近OpenBR的目的。人脸检测往往是[人脸识别](http://openbiometrics.org/docs/tutorials/#face-recognition)的第一步。我们来执行一下OpenBR中的人脸检测。打开终端输入：

```
$ br -gui -algorithm "Cvt(Gray)+Cascade(FrontalFace)+Draw(lineThickness=3)+Show(false)" -enroll 0.webcam
```

你的摄像头应该再一次被打开了，但是这一次在里的脸部多了一个框。我们添加了两个新的Transform：[Cascade](http://openbiometrics.org/docs/plugin_docs/metadata/#cascadetransform)和[Draw](http://openbiometrics.org/docs/plugin_docs/gui/#drawtransform)。我们来通过一个个的Transform来看看它是如何工作的：

- [Cvt(Gray)](http://openbiometrics.org/docs/plugin_docs/imgproc/#cvttransform)：将图像从BGR转换为灰度图。灰度图是[Cascade](http://openbiometrics.org/docs/plugin_docs/metadata/#cascadetransform)正常工作所必需的。
- [Cascade(FrontalFace)](http://openbiometrics.org/docs/plugin_docs/metadata/#cascadetransform)：这个是对[OpenCV Cascade](http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html)分类框架的一个封装版本。它使用`FrontalFace`模型来检测正脸。
- [Draw(lineThickness=3)](http://openbiometrics.org/docs/plugin_docs/gui/#drawtransform)：获取Cascade检测到的矩形框并且画到摄像头图像帧中。`lineThickness`决定了矩形框的厚度。
- [Show(false)](http://openbiometrics.org/docs/plugin_docs/gui/#showtransform)：在GUI窗口中显示图像。`false`指明图像的显示不需要等待按键操作。

每一个[Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)完成一个任务并且将其输出传递给另一个[Transform](http://openbiometrics.org/docs/api_docs/cpp_api/transform/transform/)。你可以随心所愿地连接任意多的Transform，但是要注意特定的Transform对它们的输入有着特定的需求。

你可能会思考，到底哪些对象被传递给了算法管道。在OpenBR中有两个对象来处理数据：

- [File](http://openbiometrics.org/docs/api_docs/cpp_api/file/file/)s通常被用来存储磁盘上相关元数据文件的路径信息（键值对形式）。在上面的例子中，我们将Cascade检测到的矩形框作为一个元数据，然后交给Draw来可视化。
- [Template](http://openbiometrics.org/docs/api_docs/cpp_api/template/template/)s是图像和[File](http://openbiometrics.org/docs/api_docs/cpp_api/file/file/)s的容器。图像在OpenBr中是OpenCV Mats，且是Templates的成员变量。Templates可以包含一个或多个图像。

如果你想学习更多关于[命令行](http://openbiometrics.org/docs/api_docs/cl_api/)或者[所有的插件以及关键数据结构](http://openbiometrics.org/docs/api_docs/cpp_api/)的内容，请参考链接的文档。下一篇教程将会更加深入地探讨算法以及它们的使用。



