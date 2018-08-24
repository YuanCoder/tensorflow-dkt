#-*- coding:utf-8 _*-  
""" 
@desc: 
@version: python3.5
@author: Jenpin
@file: test.py 
@time: 2018/8/21 0021 09:57
@email: yuan_268311@163.com
"""
import argparse

import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from src.TensorFlowDKT import *
from src.data_process import *

"""
声明variable和op
初始化op声明
"""
def run(args):
    # 创建saver 对象
    saver = tf.train.import_meta_graph("../data/model_save/studentTrainModel-5.meta")

    # process data
    seqs_by_student, num_skills = read_file(args.dataset)
    train_seqs, test_seqs = split_dataset(seqs_by_student ,1)
    batch_size = 10
    test_generator = DataGenerator(test_seqs, batch_size=batch_size, num_skills=num_skills)

    # config and create model
    config = {"hidden_neurons": [200],
              "batch_size": batch_size,
              "keep_prob": 0.6,
              "num_skills": num_skills,
              "input_size": num_skills * 2}
    model = TensorFlowDKT(config)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())  # 可以执行或不执行，restore的值会override初始值
        saver.restore(sess, "../data/model_save/studentTrainModel-5")
        sess.run(tf.global_variables_initializer())

        # test  测试
        test_generator.reset()
        preds, binary_preds, targets = list(), list(), list()
        while not test_generator.end:
            input_x, target_id, target_correctness, seqs_len, max_len = test_generator.next_batch()
            binary_pred, pred, _ = model.step(sess, input_x, target_id, target_correctness, seqs_len, is_train=False)
            for seq_idx, seq_len in enumerate(seqs_len):
                preds.append(pred[seq_idx, 0:seq_len])
                binary_preds.append(binary_pred[seq_idx, 0:seq_len])
                targets.append(target_correctness[seq_idx, 0:seq_len])
        # compute metrics
        preds = np.concatenate(preds)
        binary_preds = np.concatenate(binary_preds)
        targets = np.concatenate(targets)
        auc_value = roc_auc_score(targets, preds)  # 直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略。
        accuracy = accuracy_score(targets,
                                  binary_preds)  # 分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。
        precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds)  # 计算精确度、召回率、f、支持率
        print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train dkt model")
    arg_parser.add_argument("--dataset", dest="dataset", required=True)
    args = arg_parser.parse_args()
    run(args)