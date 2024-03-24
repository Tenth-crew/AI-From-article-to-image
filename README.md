## 1.项目设计
### 1.1 项目背景和目标
在如今的数字化时代，大量的文本内容被创建和共享。然而，有时候文字无法直观地传达信息，而图像则能够更好地传达复杂的概念和情感。本项目旨在利用人工智能技术，将给定的文章转化为相应的图像，以提供更直观和易于理解的信息传递方式。
### 2 技术架构
#### 2.1 摘要生成
摘要生成使用的模型是 BART (Bidirectional and Auto-Regressive Transformers)。BART 是一种序列到序列的 Transformer 模型。在实际过程中，我们通过使用 HUgging Face 的“ BartForConditionalGeneration ”类加载了一个预训练的 BART 模型，然后对给定的文章进行摘要生成。
#### 2.2 图片生成
本项目使用了稳定扩散（Stable Diffusion）模型来生成图像。
模型选择了DPMSolverMultistepScheduler来作为稳定扩散模型的调度器。

![image.png](https://cdn.nlark.com/yuque/0/2024/png/39023202/1711167956115-0b6efed3-0ef9-4b72-ac01-9a02f9a002a3.png#averageHue=%23e8cca8&clientId=u50a7ec6c-5997-4&from=paste&height=657&id=u7bdbd575&originHeight=657&originWidth=1572&originalType=binary&ratio=1&rotation=0&showTitle=false&size=154334&status=done&style=none&taskId=u222f2e91-6ebe-4feb-969b-dc8ffe3910c&title=&width=1572)

#### 2.3 优化与加载
使用**Intel**提供的**Bigdl库**低内存模式（low memory mode）来初始化Bart模型，以节省内存。加载了预训练的Bart模型的低位参数。使得运行速度得以提升。
#### 2.4 工作流程
针对文章数据Article进行文本总结得到Summarize，得到图片生成模型可以理解的 prompt 从而是的图片可以被顺利生成。

![image.png](https://cdn.nlark.com/yuque/0/2024/png/39023202/1711170405210-6ba5cf0c-5376-43fa-88b1-e99ccd85bcee.png#averageHue=%23f7f7f7&clientId=u50a7ec6c-5997-4&from=paste&height=460&id=u6ecf219e&originHeight=460&originWidth=696&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24533&status=done&style=none&taskId=ub7af935a-e335-424d-9e85-ffd5115127f&title=&width=696)

### 3 应用方向
主要应用方向为文章总结以及文生图，为新闻或者其他文字内容产生合适的配图。例如我们为宇航员在太空骑马进行了配图模拟，模拟结果如下。

![astronaut_rides_horse.png](https://cdn.nlark.com/yuque/0/2024/png/39023202/1711170843079-a83c85d4-dc9b-4a2f-b195-ec8f0b4c2890.png#averageHue=%23886349&clientId=u50a7ec6c-5997-4&from=drop&id=uc93bb845&originHeight=512&originWidth=512&originalType=binary&ratio=1&rotation=0&showTitle=false&size=401120&status=done&style=none&taskId=u758de0e1-4c35-4821-b7c1-2bce5d73fe1&title=)

### 4 Intel加速
我们通过使用**Intel**提供的**Bigdl库**低内存模式（low memory mode）来初始化Bart模型，以节省内存。加载了预训练的Bart模型的低位参数。使得运行速度得以提升。以下是我们使用程序运行的演示结果，运行时间24min。
首先我们输入如下文章

*Trump’s lawyers acknowledged Monday that he was struggling to find an insurance company willing to underwrite his $454 million bond. Privately, Trump had been counting on Chubb, which underwrote his $91.6 million bond to cover the E. Jean Carroll judgment, to come through, but the insurance giant informed his attorneys in the last several days that that option was off the table.*
*Trump’s team has sought out wealthy supporters and weighed what assets could be sold – and fast. The presumptive GOP presidential nominee himself has become increasingly concerned about the optics the March 25 deadline could present – especially the prospect that someone whose identity has long been tied to his wealth would confront financial crisis. Trump has continued to privately lash out at the New York Attorney General Letitia James and Judge Arthur Engoron over the matter, these sources told CNN.*
*Shortly before 6:30 a.m. Tuesday, Trump took those grievances public, posting on his social media platform eight times within two hours about the deadline, arguing that he shouldn’t have to put up the money and worrying that he “would be forced to mortgage or sell Great Assets, perhaps at Fire Sale prices, and if and when I win the Appeal, they would be gone.”*
*“Does that make sense? WITCH HUNT. ELECTION INTERFERENCE!” the former president wrote.*
*“These baseless innuendos are pure bullsh*t,” Trump campaign spokesman Steven Cheung said in a statement Tuesday. “President Trump has filed a motion to stay the unjust, unconstitutional, un-American judgment from New York Judge Arthur Engoron in a political Witch Hunt brought by a corrupt Attorney General. A bond of this size would be an abuse of the law, contradict bedrock principals of our Republic, and fundamentally undermine the rule of law in New York.*

得到如下摘要和图片。

![b14ed6a16f9a441377523f9572af1804.png](https://cdn.nlark.com/yuque/0/2024/png/39023202/1711181949497-e2244ba9-757e-4113-ae58-a21a18e72006.png#averageHue=%23966f56&clientId=u0637c5e0-7f23-4&from=drop&id=uaa754c8b&originHeight=1010&originWidth=714&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1155849&status=done&style=none&taskId=u0456bfae-7d70-484b-ae10-46de659638a&title=)

### 5 使用方法

1.安装streamlit包

2.安装其余缺失包

3.通过 streamlit run streamlit.py 这个命令启动文件

