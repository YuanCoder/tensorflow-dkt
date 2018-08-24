import argparse
import time
import sys
from src.TensorFlowDKT import *
from src.data_process import *
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

model_path = "../data/model_save/studentTrainModel"
global_step = 5

def run(args):
    ## training model
    if args.mode == 'train':
        # process data
        seqs_by_student, num_skills = read_file(args.dataset)
        train_seqs, test_seqs = split_dataset(seqs_by_student)
        batch_size = 10
        train_generator = DataGenerator(train_seqs, batch_size=batch_size, num_skills=num_skills)
        test_generator = DataGenerator(test_seqs, batch_size=batch_size, num_skills=num_skills)

        # config and create model
        config = {"hidden_neurons": [200],
                  "batch_size": batch_size,
                  "keep_prob": 0.6,
                  "num_skills": num_skills,
                  "input_size": num_skills * 2}
        model = TensorFlowDKT(config)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        lr = 0.4 #学习率
        lr_decay = 0.92 #学习率衰减
        # run epoch
        for epoch in range(10):
            # train
            model.assign_lr(sess, lr * lr_decay ** epoch) #assign_lr 方法来设置学习率
            overall_loss = 0
            train_generator.shuffle()
            st = time.time()
            # 训练
            while not train_generator.end:
                input_x, target_id, target_correctness, seqs_len, max_len = train_generator.next_batch()
                overall_loss += model.step(sess, input_x, target_id, target_correctness, seqs_len, is_train=True)
                print ("\r idx:{0}, overall_loss:{1}, time spent:{2}s".format(train_generator.pos, overall_loss, time.time() - st))
                sys.stdout.flush()
            if(train_generator.end):
                # 保存模型

                saver = tf.train.Saver()
                # model_path = "../data/model_save/testModel";
                saver.save(sess, model_path, global_step)
                saver.save(sess, "../data/model_save/studentTrainModel.ckpt", global_step)
                print("模型已保存！")

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
            auc_value = roc_auc_score(targets, preds) #直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略。
            accuracy = accuracy_score(targets, binary_preds) #分类准确率分数是指所有分类正确的百分比。分类准确率这一衡量分类器的标准比较容易理解，但是它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。
            precision, recall, f_score, _ = precision_recall_fscore_support(targets, binary_preds) #计算精确度、召回率、f、支持率
            print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision, recall))

    elif args.mode == 'test':
        seqs_by_student, num_skills = read_file(args.dataset)
        train_seqs, test_seqs = split_dataset(seqs_by_student)
        batch_size = 10
        test_generator = DataGenerator(test_seqs, batch_size=batch_size, num_skills=num_skills)
        # config and create model
        config = {"hidden_neurons": [200],
                  "batch_size": batch_size,
                  "keep_prob": 0.6,
                  "num_skills": num_skills,
                  "input_size": num_skills * 2}
        model = TensorFlowDKT(config)

        sess = tf.Session()
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
        precision, recall, f_score, _ = precision_recall_fscore_support(targets,
                                                                        binary_preds)  # 计算精确度、召回率、f、支持率
        print("\n auc={0}, accuracy={1}, precision={2}, recall={3}".format(auc_value, accuracy, precision,
                                                                           recall))

        print("模型路径{}:".format(model_path))
        ckpt_file = tf.train.latest_checkpoint("../data/model_save/")
        print("ckpt文件{}:".format(ckpt_file))
        # with tf.Session() as sess:
        #     saver = tf.train.import_meta_graph('../data/model_save/studentTrainModel-5.meta')
        #     saver.restore(sess, tf.train.latest_checkpoint(ckpt_file))
        #     print("加载模型!")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="train dkt model")
    arg_parser.add_argument("--dataset", dest="dataset", required=True)
    arg_parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
    args = arg_parser.parse_args()
    run(args)