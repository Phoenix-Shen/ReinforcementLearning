# MADDPG

代码参考了这两个版本：

[XUEHY MADDPG](https://github.com/xuehy/pytorch-maddpg) 和 [Starry-sky6688 MADDPG](https://github.com/starry-sky6688/MADDPG)

具体推导可以看本仓库根目录readme文件的MADDPG部分。

## 建议重新创个环境跑

pip install gym==0.10.5

由于多Agent环境太老了，需要gym低版本，所以推荐重新创个环境跑。

## 优化思路

多Agent需要执行for循环，能否把它变成多线程呢，这样就可以大大加快速度了。
