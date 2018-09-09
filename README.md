# SparseCNNs

本文主要研究了稀疏输入下的卷积神经网络，并用稀疏的激光雷达扫描数据进行实验。传统的CNN在输入稀疏数据时性能很差，即使提供了丢失数据的位置，效果也不理想。为了解决这个问题，本文提出了一个简单有效的稀疏卷积层，在卷积运算中明确考虑了丢失数据的位置。
