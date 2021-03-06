import numpy as np
import random

'''
读取学生做题记录文件
@dataset_path 做题路径
'''
def read_file(dataset_path):
    seqs_by_student = {}
    num_skills = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            student, problem, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
            num_skills = max(num_skills, problem)
            seqs_by_student[student] = seqs_by_student.get(student, []) + [[problem, is_correct]]
    return seqs_by_student, num_skills + 1


def split_dataset(seqs_by_student, sample_rate=0.5, random_seed=1):
    sorted_keys = sorted(seqs_by_student.keys())
    random.seed(random_seed) #random.seed(0)作用：使得随机数据可预测，即只要seed的值一样，后续生成的随机数都一样。
  #  test_keys = random.sample(sorted_keys, int(len(sorted_keys) * sample_rate))
    test_keys = set(random.sample(sorted_keys, int(len(sorted_keys) * sample_rate))) #从指定序列中随机获取指定长度的片断    sample_rate 抽样率
    test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys] # 随机学号对应的题目
    train_seqs = [seqs_by_student[k] for k in seqs_by_student if k not in test_keys] # 不在随机抽取学号对应的题目
    return train_seqs, test_seqs

#填充队列
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence 从第一个非空序列中取样
    # checking for consistency in the main loop below.  检查下面主循环的一致性
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape 检查`trunc`有预期的形状
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# 转换成one-hot
def num_to_one_hot(num, dim):
    base = np.zeros(dim)
    if num >= 0:
        base[num] += 1
    return base

#格式化数据
def format_data(seqs, batch_size, num_skills):
    gap = batch_size - len(seqs)
    seqs_in = seqs + [[[0, 0]]] * gap  # pad batch data to fix size
    seq_len = np.array(list(map(lambda seq: len(seq), seqs_in))) - 1
    max_len = max(seq_len) #最大问题长度
    x = pad_sequences(np.array([[(j[0] + num_skills * j[1]) for j in i[:-1]] for i in seqs_in]), maxlen=max_len, padding='post', value=-1) # 长度达不到 maxlen的列表用-1来填充
    input_x = np.array([[num_to_one_hot(j, num_skills*2) for j in i] for i in x]) # 题目One-Hot码
    target_id = pad_sequences( np.array([[j[0] for j in i[1:]] for i in seqs_in]), maxlen=max_len, padding='post', value=0 ) # 长度达不到 maxlen的列表用-1来填充
    target_correctness = pad_sequences(np.array([[j[1] for j in i[1:]] for i in seqs_in]), maxlen=max_len, padding='post', value=0)
    return input_x, target_id, target_correctness, seq_len, max_len
    #输入题目One-Hot向量矩阵，问题矩阵，   答案矩阵，    队列长度，最大长度


class DataGenerator(object):
    def __init__(self, seqs, batch_size, num_skills):
        self.seqs = seqs
        self.batch_size = batch_size
        self.pos = 0
        self.end = False
        self.size = len(seqs)
        self.num_skills = num_skills

    def next_batch(self):
        batch_size = self.batch_size
        if self.pos + batch_size < self.size:
            batch_seqs = self.seqs[self.pos:self.pos + batch_size]
            self.pos += batch_size
        else:
            batch_seqs = self.seqs[self.pos:]
            self.pos = self.size - 1
        if self.pos >= self.size - 1:
            self.end = True
        input_x, target_id, target_correctness, seqs_len, max_len = format_data(batch_seqs, batch_size, self.num_skills)
        return input_x, target_id, target_correctness, seqs_len, max_len

    #随机序列
    def shuffle(self):
        self.pos = 0
        self.end = False
        np.random.shuffle(self.seqs)
    #重置
    def reset(self):
        self.pos = 0
        self.end = False
