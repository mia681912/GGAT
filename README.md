# GGAT
High-frequency geometry enhanced graph attention network for hyperspectral and multispectral image fusion

# Dataset
CAVE, Harvard, Pavia Center, Ziyuan-01

CAVE: http://www.cs.columbia.edu/CAVE/databases/

How to make .mat dataset can refer to my blog:
https://blog.csdn.net/fgjnghh/article/details/136590072?spm=1001.2014.3001.5502

Harvard: http://vision.seas.harvard.edu/hyperspec/

Pavia Center: http://www.ehu.eus/ccwintco/index.php

Ziyuan-01 real HS-MS data: https://github.com/rs-lsl/CSSNet

# Train the model
python train.py --batch_size=2 

# Test the model
python test.py

# Comparative model link:

IR-TenSR: https://github.com/liangjiandeng/IR_TenSR

SSR-Net: https://github.com/hw2hwei/SSRNET

PZRes-Net: https://github.com/zbzhzhy/PZRes-Net

GuideNet: https://github.com/Evangelion09/GuidedNet

MOG-DCN: https://see.xidian.edu.cn/faculty/wsdong/Projects/MoG-DCN.htm

UAL: https://github.com/JiangtaoNie/UAL

SDAGE: https://github.com/RSMagneto/SDAGE

The parameter values can refer to the experimental section of the original text.
