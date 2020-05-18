目录：

文件夹
----------------------------------------
OPP-115：OPP-115语料库原始文件
original_documents：APP-350语料库原始html文件
由于文件太大，这里两个文件夹为空文件，可以在www.usableprivacy.org下载

result：不同embedding参数及最大句子长度下分类器的得分

文件
----------------------------------------

clean_data: 经处理后用于训练的OPP-115语料
	clean_data_0.5： 0.5代表阈值，自行处理
	clean_data_0.5_no_repeat：为去重后的文件，训练使用该文件
	clean_data_0.5_no_attribute: 为未去重但去除了各类别属性仅保留类别的文件

10_run_average: 各神经网络算法跑10轮取平均
get_best_score: 不同参数调参取最好结果
SVM_Doc2vec&Word2vec：svm用deoc2vec与word2vec对比
Word2Vec：训练Word2Vec向量
Doc2Vec：训练Doc2Vec向量

OPP115_data_generation：从原始语料库生成训练用数据
load_data：加载训练数据
count_data：统计文本长度及类别数量
