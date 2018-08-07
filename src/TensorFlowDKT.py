# encoding:utf-8
import tensorflow as tf

'''
模型构建
'''
class TensorFlowDKT(object):
    def __init__(self, config):
        self.hidden_neurons = hidden_neurons = config["hidden_neurons"]
        self.num_skills = num_skills = config["num_skills"] #题库的题目数量
        self.input_size = input_size = config["input_size"] #输入层节点数，等于题库数量*2
        self.batch_size = batch_size = config["batch_size"]
        self.keep_prob_value = config["keep_prob"]

        '''
        一、首先初始化模型参数，并且用tf.placeholder来接收模型的输入
            这里我们用 tf.dynamic_rnn 构建了一个多层循环神经网络，cell 参数用来指定了隐层神经元的结构，sequence_len 参数表示一个 batch 中各个序列的有效长度。
            state_series 表示隐层的输出，是一个三阶的 Tensor，self.current_state 表示 batch 各个序列的最后一个 step 的隐状态。
        '''
        # 接收输入
        self.max_steps = tf.placeholder(tf.int32)  # max seq length of current batch
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size]) #答题信息
        self.sequence_len = tf.placeholder(tf.int32, [batch_size])  #一个batch中每个序列的有效长度
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob

        # 接收标签信息
        self.target_id = tf.placeholder(tf.int32, [batch_size, None])   #回答题目的ID
        self.target_correctness = tf.placeholder(tf.float32, [batch_size, None]) #答题对错情况

        # create rnn cell
        '''
        二、构建RNN层
            这里我们用 tf.dynamic_rnn 构建了一个多层循环神经网络，cell 参数用来指定了隐层神经元的结构，sequence_len 参数表示一个 batch 中各个序列的有效长度。
            state_series 表示隐层的输出，是一个三阶的 Tensor，self.current_state 表示 batch 各个序列的最后一个 step 的隐状态。
        '''
        hidden_layers = []
        for idx, hidden_size in enumerate(hidden_neurons):
            lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer,
                                                         output_keep_prob=self.keep_prob)
            hidden_layers.append(hidden_layer)
        self.hidden_cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

        # dynamic rnn
        state_series, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                             inputs=self.input_data,
                                                             sequence_length=self.sequence_len,
                                                             dtype=tf.float32)
        '''
        三、输出层
            输出层我们构建了两个变量作为隐层到输出层的连接的参数，并用 tf.sigmoid 作为激活函数。
            到这里我们已经可以得到模型的预测输出 self.pred_all，这也是一个三阶的张量，shape 为(batch_size, self.max_steps, num_skills)。
        '''
        # output layer
        output_w = tf.get_variable("W", [hidden_neurons[-1], num_skills])
        output_b = tf.get_variable("b", [num_skills])
        self.state_series = tf.reshape(state_series, [batch_size * self.max_steps, hidden_neurons[-1]])
        self.logits = tf.matmul(self.state_series, output_w) + output_b
        self.mat_logits = tf.reshape(self.logits, [batch_size, self.max_steps, num_skills])
        self.pred_all = tf.sigmoid(self.mat_logits)

        '''
        为了训练模型，还需要计算模型损失函数和梯度，我们结合预测和标签信息来获得损失函数
        '''
        # compute loss
        flat_logits = tf.reshape(self.logits, [-1])
        flat_target_correctness = tf.reshape(self.target_correctness, [-1])
        flat_base_target_index = tf.range(batch_size * self.max_steps) * num_skills
        flat_bias_target_id = tf.reshape(self.target_id, [-1])
        flat_target_id = flat_bias_target_id + flat_base_target_index
        flat_target_logits = tf.gather(flat_logits, flat_target_id)
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [batch_size, self.max_steps]))
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.int32)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness,
                                                                          logits=flat_target_logits))
        '''
        获得梯度并更新参数
        '''
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 4)

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.grads, trainable_vars))

    '''
    需要注意的是，在用 tf.gradients 得到梯度后，我们使用了 tf.clip_by_global_norm 方法，这主要是为了防止梯度爆炸的现象。
    最后应用了一次梯度下降得到的 self.train_op 就是计算图的训练结点。得到训练结点后，我们的计算图 (Graph) 就已经构造完毕，
    接着只需要创建一个 tf.Session 对象，并调用其run()方法来运行计算图就可以进行模型训练和测试了。由于训练和测试的接收的feed_dict类似，我们定义 step 方法来用作训练和测试
    '''
    # step on batch
    def step(self, sess, input_x, target_id, target_correctness, sequence_len, is_train):
        _, max_steps, _ = input_x.shape
        input_feed = {self.input_data: input_x,
                      self.target_id: target_id,
                      self.target_correctness: target_correctness,
                      self.max_steps: max_steps,
                      self.sequence_len: sequence_len}

        if is_train:
            input_feed[self.keep_prob] = self.keep_prob_value
            train_loss, _, _ = sess.run([self.loss, self.train_op, self.current_state], input_feed)
            return train_loss
        else:
            input_feed[self.keep_prob] = 1
            bin_pred, pred, pred_all = sess.run([self.binary_pred, self.pred, self.pred_all], input_feed)
            return bin_pred, pred, pred_all

    '''
    定义 assign_lr 方法来设置学习率:
    '''
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))