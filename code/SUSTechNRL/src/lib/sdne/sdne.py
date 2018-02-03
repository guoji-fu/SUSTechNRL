import numpy as np
import tensorflow as tf
import random
import copy

from .rbm import *
from .utils import *

class SDNE:
    def __init__(self, 
                 graph,
                 struct,
                 alpha,
                 gamma,
                 reg,
                 beta,
                 batch_size,
                 epochs_limit,
                 learning_rate,
                 display,
                 DBN_init,
                 dbn_epochs,
                 dbn_batch_size,
                 dbn_learning_rate,
                 sparse_dot,
                 negative_ratio,
                 dim,
                 output_file):
        self.graph = graph
        self.struct = struct
        self.alpha = alpha
        self.gamma = gamma
        self.reg = reg
        self.beta = beta
        self.batch_size = batch_size
        self.epochs_limit = epochs_limit
        self.learning_rate = learning_rate
        self.display = display
        self.DBN_init = DBN_init
        self.dbn_epochs = dbn_epochs
        self.dbn_batch_size = dbn_batch_size
        self.dbn_learning_rate = dbn_learning_rate
        self.sparse_dot = sparse_dot
        self.negative_ratio = negative_ratio
        self.dim = dim
        self.output_file = output_file

        self.struct[0] = self.graph.node_size
        self.struct[2] = self.dim

        # print('number of nodes: ', self.graph.node_size)
        # print('number of edges: ', self.graph.G.number_of_edges())

        self.order = np.arange(self.graph.node_size)
        self.is_epoch_end = False
        self.st = 0
        self.embedding = None
        self.vectors = {}

        self.is_variables_init = False
        ######### not running out gpu sources ##########
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config =  tf_config)

        ############ define variables ##################
        self.layers = len(self.struct)
        self.W = {}
        self.b = {}
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        self.struct.reverse()
        ###############################################
        ############## define input ###################
                
        self.adjacent_matriX = tf.placeholder("float", [None, None])
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64)
        self.X_sp_ids_val = tf.placeholder(tf.float32)
        self.X_sp_shape = tf.placeholder(tf.int64)
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        #
        self.X = tf.placeholder("float", [None, self.struct[0]])
        
        ###############################################
        self.__make_compute_graph()
        self.loss = self.__make_loss()
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        
        self.train()
    
    def __make_compute_graph(self):
        def encoder(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        def encoder_sp(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                if i == 0:
                    X = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X, self.W[name]) + self.b[name])
                else:
                    X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        def decoder(X):
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        if self.sparse_dot:
            self.H = encoder_sp(self.X_sp)
        else:
            self.H = encoder(self.X)
        self.X_reconstruct = decoder(self.H)
    

        
    def __make_loss(self):
        def get_1st_loss_link_sample(self, Y1, Y2):
            return tf.reduce_sum(tf.pow(Y1 - Y2, 2))
        def get_1st_loss(H, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch,1))
            L = D - adj_mini_batch ## L is laplation-matriX
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X)* B, 2))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.itervalues()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
            return ret
            
        #Loss function
        self.loss_2nd = get_2nd_loss(self.X, self.X_reconstruct, self.beta)
        self.loss_1st = get_1st_loss(self.H, self.adjacent_matriX)
        self.loss_xxx = tf.reduce_sum(tf.pow(self.X_reconstruct,2)) 
        # we don't need the regularizer term, since we have nagetive sampling.
        #self.loss_reg = get_reg_loss(self.W, self.b) 
        #return self.gamma * self.loss_1st + self.alpha * self.loss_2nd + self.reg * self.loss_reg
        
        return self.gamma * self.loss_1st + self.alpha * self.loss_2nd +self.loss_xxx

    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        self.is_Init = True
    
    def do_variables_init(self, data, DBN_init):
        def assign(a, b):
            op = a.assign(b)
            self.sess.run(op)
        init = tf.global_variables_initializer()       
        self.sess.run(init)
        if DBN_init:
            shape = self.struct
            myRBMs = []
            for i in range(len(shape) - 1):
                myRBM = rbm([shape[i], shape[i+1]], {"batch_size": self.dbn_batch_size, "learning_rate":self.dbn_learning_rate})
                myRBMs.append(myRBM)
                for epoch in range(self.dbn_epochs):
                    error = 0
                    for batch in range(0, data.G.number_of_nodes(), self.dbn_batch_size):
                        mini_batch = self.sample(self.dbn_batch_size).X
                        for k in range(len(myRBMs) - 1):
                            mini_batch = myRBMs[k].getH(mini_batch)
                        error += myRBM.fit(mini_batch)
                    print("rbm epochs:", epoch, "error : ", error)

                W, bv, bh = myRBM.getWb()
                name = "encoder" + str(i)
                assign(self.W[name], W)
                assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                assign(self.W[name], W.transpose())
                assign(self.b[name], bv)
        self.is_Init = True

    def __get_feed_dict(self, data):
        X = data.X
        if self.sparse_dot:
            X_ind = np.vstack(np.where(X)).astype(np.int64).T
            X_shape = np.array(X.shape).astype(np.int64)
            X_val = X[np.where(X)]
            return {self.X : data.X, self.X_sp_indices: X_ind, self.X_sp_shape:X_shape, self.X_sp_ids_val: X_val, self.adjacent_matriX : data.adjacent_matriX}
        else:
            return {self.X: data.X, self.adjacent_matriX: data.adjacent_matriX}
            
    def fit(self, data):
        feed_dict = self.__get_feed_dict(data)
        ret, _ = self.sess.run((self.loss, self.optimizer), feed_dict = feed_dict)
        return ret
    
    def get_loss(self, data):
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.H, feed_dict = self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)
        
    def get_B(self):
        return self.sess.run(self.b)
        
    def close(self):
        self.sess.close()

    def sample(self, do_shuffle=True):
        node_size = self.graph.node_size
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self.order[0:node_size])
            else:
                self.order = np.sort(self.order)
            self.st = 0
            self.is_epoch_end = False
        mini_batch = Dotdict()
        en = min(node_size, self.st + self.batch_size)
        index = self.order[self.st:en]
        mini_batch.X = self.graph.adjMat[index]
        mini_batch.adjacent_matriX = self.graph.adjMat[index][:][:,index]
        if en == node_size:
            en = 0
            self.is_epoch_end = True
        self.st = en
        
        return mini_batch
    
    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        fout.write("{} {}\n".format(self.graph.node_size, self.dim))
        node = 0
        for vec in self.embedding:
            fout.write("{} {}\n".format(str(node),
                                        ' '.join([str(x) for x in vec])))
            self.vectors[str(node)] = vec
            node += 1
        fout.close()

    def train(self):
        self.do_variables_init(self.graph, self.DBN_init)

        epochs = 0
        batch_n = 0

        origin_data = copy.deepcopy(self.graph)
        # fout = open(self.output_file, 'w')
        while True:
            mini_batch = self.sample(self.batch_size)
            loss = self.fit(mini_batch)
            batch_n += 1
            print('Epoch: %d, batch: %d, loss: %.3f' % (epochs, batch_n, loss))
            if self.is_epoch_end:
                epochs += 1
                batch_n = 0
                print('Epoch: %d loss: %.3f' % (epochs, loss))
                if epochs % self.display == 0:
                    embedding = None
                    while True:
                        mini_batch = self.sample(do_shuffle=False)
                        print('--------------')
                        print(mini_batch)
                        print('--------------')
                        loss += self.get_loss(mini_batch)
                        if embedding is None:
                            embedding = self.get_embedding(mini_batch)
                        else:
                            embedding = np.vstack((embedding, self.get_embedding(mini_batch)))

                        if self.is_epoch_end:
                            break
                    
                    # result = check_link_reconstruction(embedding, self.graph, [20000,40000,60000,80000,100000])
                    # print(epochs, result, file=fout)
                    # sio.savemat('SDNE_embedding' + '-' + str(epochs) + '.mat', {'embedding': embedding})
                if epochs > self.epochs_limit:
                    print('exceed epochs limit terminating')
                    break
                last_loss = loss
                
                self.embedding = embedding
                # print('--------------------')
                # print(str(embedding))
                # print('--------------------')
                # print(np.shape(embedding))
                # print('--------------------')

            # fout.close()