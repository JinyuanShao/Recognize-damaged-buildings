# BDD-Net
A deep learning model that recognize damaged buildings for based on post-disaster satellite imagery.

The code would release soon.

疫情过后上传代码（还不能回学校）

主要思路是以EfficientNet-B0作为backbone构建一个U-Net结构的神经网络进行灾后建筑物的语义分割。

主要特点是模型接收灾前和灾后两个时间的图像，并融合为一个6通道的新张量放入模型，实验证明只输入灾后图像网络效果差，很难辨别建筑物，因为灾后场景往往建筑群被夷为平地，无法从灾后图像提取有用信息。如果输入6通道，将大大提高分割结果。

在训练过程中，为了更好的优化建筑物像素的类别不平衡，采用了Dice+focal结合的loss，实验证明该loss对于区分受损和完好建筑物是有效的。
