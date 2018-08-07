
### Python3运行环境    
### 数据说明 
### /data/assistments.tx 里一行对应的格式是学生id，题目id，问题对错
### 模型
模型的输入序列x1,x2,x3…对应了t1,t2,t3…时刻学生答题信息的编码，隐层状态对应了各个时刻学生的知识点掌握情况，模型的输出序列对应了各时刻学生回答题库中的所有习题答对的概率。

`$python train_dkt.py --dataset ../data/assistments.txt`
