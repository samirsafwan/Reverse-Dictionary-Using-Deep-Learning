import os, random
import tensorflow as tf
import numpy as np
import loadData, params, util
from model import Model
import logging
import scipy.spatial.distance as ssd
import attentiontf

class LSTM_Model(Model):
    def __init__(self, word_actualvec_dict, meaningwordslist_word_list, metric_type):
        super().__init__(word_actualvec_dict, meaningwordslist_word_list, metric_type)
        self.model = params.MODEL_LSTM

    def train(self):
        '''
        train model and populate self.trainedvec_word_dict
        '''
        #inputs = tf.placeholder(tf.float32, (None, None, params.INPUT_SIZE))  # (seq_len, batch, in)
        #outputs = tf.placeholder(tf.float32, (None, params.OUTPUT_SIZE))  # (1, batch, out)
        inputs = tf.placeholder(tf.float32, (None, None, params.INPUT_SIZE), name="inputs")  # (seq_len, batch, in)
        outputs = tf.placeholder(tf.float32, (None, params.OUTPUT_SIZE), name="outputs")  # (1, batch, out)

        num_units = params.HIDDEN_LAYERS
        batch_size = tf.shape(inputs)[1]

        out_weights = tf.Variable(tf.random_normal([num_units, params.INPUT_SIZE]), name="outweights")

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, time_major=True)
        last_state = rnn_outputs[-1]
        context_vector = tf.matmul(last_state, out_weights)

        if params.ATTENTION == True:
            attention_output, alphas = attentiontf.attention(rnn_outputs, 10)
            keep_prob = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(attention_output, keep_prob)

            W = tf.Variable(tf.truncated_normal([params.HIDDEN_LAYERS, 1], stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[1]))
            y_hat = tf.nn.xw_plus_b(dropout, W, b)
            y_hat = tf.squeeze(y_hat)

        print("inputs shape:", inputs)
        print("outputs shape:", outputs)
        print("rnn_outputs name shape:", rnn_outputs[0].name, rnn_outputs)
        print("final_states  shape:" , final_states)
        print("last output shape", last_state)
        # error = tf.norm(outputs - last_state, axis = 1)
        # print("error", error.shape)
        # error = tf.reduce_mean(error)
        # print("avg error", error.shape)
        print("context_vector name shape:", context_vector.name, context_vector)
        print("out_weights name shape:", out_weights.name, out_weights)

        # loss_function (norm)
        error = context_vector - outputs
        net = [v for v in tf.trainable_variables()]
        weight_reg = tf.add_n([params.REG_CONST * tf.nn.l2_loss(var) for var in net])
        if self.metric_type == params.METRIC_COSINE:
            # loss_function (cosine distance)
            loss = tf.losses.cosine_distance(tf.nn.l2_normalize(context_vector, 0),
                                             tf.nn.l2_normalize(outputs, 0), dim=0)  # + weight_reg

            # optimize
            train_fn = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE).minimize(loss)
            accuracy = 0
            for i in range(params.BATCH_SIZE):
                accuracy += tf.cast(tf.losses.cosine_distance(
                    tf.nn.l2_normalize(context_vector[i], 0),
                    tf.nn.l2_normalize(outputs[i], 0), dim=0) < params.METRIC_THRESHOLD, tf.float32)
        else: #norm
            loss = tf.reduce_mean(tf.pow(error, 2)) #+ weight_reg

            # optimize
            train_fn = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE).minimize(loss)
            # accuracy = tf.reduce_mean([tf.cast(tf.norm(error[i]) < params.METRIC_THRESHOLD,tf.float32)
            # for i in range(error.shape[0])])
            accuracy = 0
            for i in range(params.BATCH_SIZE):
                accuracy += tf.cast(tf.norm(error[i]) < params.METRIC_THRESHOLD, tf.float32)

        accuracy /= params.BATCH_SIZE
        ################################################################################
        ##                           DATA HYGIENE                                     ##
        ################################################################################
        training_data = util.createTrainingDataSet(self.word_actualvec_dict, self.meaningwordslist_word_list)
        random.shuffle(training_data)
        lenth = int(0.8 * len(training_data))
        training_set = training_data[:lenth]
        validation_set = training_data[lenth:]
        x_val = np.array([np.array(entry[0]) for entry in validation_set])
        y_val = np.array([np.array(entry[1]) for entry in validation_set])
        x_val = np.transpose(x_val, [1, 0, 2])
        print(training_set[0][0].shape, training_set[0][1].shape, len(training_set))
        # print(training_set[10][0].shape, training_set[10][1].shape)
        print(validation_set[0][0].shape, validation_set[0][1].shape, len(validation_set))

        ################################################################################
        ##                           TRAINING LOOP                                    ##
        ################################################################################
        saved = False
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            saver = tf.train.Saver()
            size = int(len(training_set) / params.BATCH_SIZE)
            maxval = 0
            for epoch in range(params.EPOCHS):
                random.shuffle(training_set)
                x_train = np.array([np.array(entry[0]) for entry in training_set])
                y_train = np.array([np.array(entry[1]) for entry in training_set])
                epoch_error = 0
                for it in range(params.ITERATIONS_PER_EPOCH):
                    for _ in range(size):
                        x = x_train[epoch * params.BATCH_SIZE:(epoch + 1) * params.BATCH_SIZE]
                        y = y_train[epoch * params.BATCH_SIZE:(epoch + 1) * params.BATCH_SIZE]
                        x = np.transpose(x, [1, 0, 2])
                        if params.ATTENTION == True:
                            epoch_error += session.run([loss, train_fn], {inputs: x, outputs: y, keep_prob: 0.5})[0]
                        else:
                            epoch_error += session.run([loss, train_fn], {inputs: x, outputs: y, })[0]
                    print("{0} iteration complete".format(it))
                epoch_error /= params.ITERATIONS_PER_EPOCH
                if params.ATTENTION == True:
                    valid_accuracy = session.run(accuracy, {inputs: x_val, outputs: y_val, keep_prob: 1.0})
                else:
                    valid_accuracy = session.run(accuracy, {inputs: x_val, outputs: y_val, })
                print("Epoch {0}, train error: {1}, valid accuracy: {2}".format(epoch, epoch_error,
                                                                                valid_accuracy * 100.0))
                logging.info(
                    "Epoch {0}, train error: {1}, valid accuracy: {2}".format(epoch, epoch_error,
                                                                              valid_accuracy * 100.0))
                if ((maxval == 0) and (valid_accuracy >0)) or (
                    (maxval > 0) and (valid_accuracy > maxval)):
                    maxval = valid_accuracy
                    saver.save(session, os.path.join(os.getcwd(), params.TRAINED_MODEL_PATH))
                    #self.trainedvec_word_dict = self.savetrainedvecdict()
                    saved = True
                #if saved: break
            if not saved:
                saver.save(session, os.path.join(os.getcwd(), params.TRAINED_MODEL_PATH))
                #self.trainedvec_word_dict = self.savetrainedvecdict()



    def calculateModelVec(self, input_phrase_2dvec):
        with tf.Session() as sess:
            # First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph('model/trained_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            # print([node.name for node in tf.get_default_graph().as_graph_def().node])
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN/BasicLSTMCell"):
                print(v.name)
            ops = ['MatMul:0']  # 'strided_slice_2:0','strided_slice_3:0']
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            graph = tf.get_default_graph()
            input_phrase_vector = np.reshape(input_phrase_2dvec,
                                             (input_phrase_2dvec.shape[0], 1, input_phrase_2dvec.shape[1]))
            #inputs = graph.get_tensor_by_name("Placeholder:0")
            #outputs = graph.get_tensor_by_name("Placeholder_1:0")
            inputs = graph.get_tensor_by_name("inputs:0")
            outputs = graph.get_tensor_by_name("outputs:0")
            output_vec = np.zeros((1, params.OUTPUT_SIZE))
            feed_dict = {inputs: input_phrase_vector, outputs: output_vec}
            context_vec = sess.run(ops, feed_dict)
            return context_vec  # , h1,h2

    def savedictTrained(self):
        temp =[]
        for meaningwordlist_word_tuple in self.meaningwordslist_word_list:
            wordlist = meaningwordlist_word_tuple[0]
            phrase_2dvec = util.phraselist_to_2dvec(wordlist, self.word_actualvec_dict)
            temp.append((util.condition_vector(phrase_2dvec),
                                              meaningwordlist_word_tuple[1]))
        with tf.Session() as sess:
            # First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph('model/trained_model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            # print([node.name for node in tf.get_default_graph().as_graph_def().node])
            for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope="RNN/BasicLSTMCell"):
                print(v.name)
            ops = ['MatMul:0']  # 'strided_slice_2:0','strided_slice_3:0']
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            graph = tf.get_default_graph()
            for entry in temp:
                input_phrase_2dvec = entry[0]
                input_phrase_vector = np.reshape(input_phrase_2dvec,
                                                 (input_phrase_2dvec.shape[0], 1, input_phrase_2dvec.shape[1]))
                #inputs = graph.get_tensor_by_name("Placeholder:0")
                #outputs = graph.get_tensor_by_name("Placeholder_1:0")
                inputs = graph.get_tensor_by_name("inputs:0")
                outputs = graph.get_tensor_by_name("outputs:0")
                output_vec = np.zeros((1, params.OUTPUT_SIZE))
                feed_dict = {inputs: input_phrase_vector, outputs: output_vec}
                context_vec = sess.run(ops, feed_dict)
                self.trainedvec_word_dict.append((np.array(context_vec),entry[1]))
            temp = []
