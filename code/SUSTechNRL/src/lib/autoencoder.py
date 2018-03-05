import tensorflow as tf
import numpy as np

class Autoencoder:
    def __init__(self, 
                 graph,
                 struct=[None, 200, None],
                 alpha=100,
                 gamma=1,
                 reg=1,
                 beta=10,
                 rep_size=128, 
                 epochs=500, 
                 batch_size=256,
                 learning_rate=0.001):
        self.g = graph
        self.struct = struct
        self.alpha = alpha
        self.gamma = gamma
        self.reg = reg
        self.beta = beta
        self.input_dim = np.shape(self.g.adjMat)[1]
        self.hidden_dim = rep_size
        self.epochs = epochs
        self.batch_size = batch_size
        # self.batch_size = self.g.node_size
        self.learning_rate = learning_rate

        self.struct[0] = self.g.node_size
        self.struct[-1] =rep_size
        self.embedding = None
        self.vectors = {}
        self.layers = len(self.struct)
        self.W = {}
        self.b = {}

        struct = self.struct

        x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim])

        with tf.name_scope('encode'):
            for i in range(self.layers-1):
                name = 'encoder' + str(i)
                self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]], dtype=tf.float32), name=name)
                self.b[name] = tf.Variable(tf.zeros([struct[i+1]], dtype=tf.float32), name=name)
            encoded = self.encoder(x)
        struct.reverse()
        with tf.name_scope('decode'):
            for i in range(self.layers-1):
                name = 'decoder' + str(i)
                self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]], dtype=tf.float32), name=name)
                self.b[name] = tf.Variable(tf.zeros([struct[i+1]], dtype=tf.float32), name=name)
            decoded = self.decoder(encoded)
        
        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        # self.loss = tf.reduce_sum(tf.pow(tf.subtract(self.x, self.decoded), 2))
        self.loss = self.all_loss()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def encoder(self, X):
        for i in range(self.layers-1):
            name = 'encoder' + str(i)
            X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
        
        return X
    
    def decoder(self, X):
        for i in range(self.layers-1):
            name = 'decoder' + str(i)
            X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])

        return X
    
    def all_loss(self):
        def get_1st_loss(H, adjmat):
            D = tf.diag(tf.reduce_sum(adjmat, 1))
            L = D - adjmat
            
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H), L), H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            
            return tf.reduce_sum(tf.pow(tf.subtract(self.x, self.decoded)*B, 2))

        def get_reg_loss(weights, biases):
            ret1 = 0
            ret2 = 0
            for i in range(self.layers-1):
                name1 = 'encoder' + str(i)
                name2 = 'decoder' + str(i)
                ret1 = ret1 + tf.nn.l2_loss(weights[name1]) + tf.nn.l2_loss(weights[name2])
                ret2 = ret2 + tf.nn.l2_loss(biases[name1]) + tf.nn.l2_loss(biases[name2])
            ret = ret1 + ret2

            return ret
        
        # self.loss_1st = get_1st_loss(self.encoded, self.x)
        self.loss_2nd = get_2nd_loss(self.x, self.decoded, self.beta)
        self.loss_xxx = tf.reduce_sum(tf.pow(self.decoded, 2))
        self.loss_reg = get_reg_loss(self.W, self.b) 

        # return self.gamma*self.loss_1st + self.alpha*self.loss_2nd + self.loss_reg + self.loss_xxx
        return self.loss_2nd + self.loss_reg + self.loss_xxx
    
    def get_batch(self, X, batch_size):
        a = np.random.choice(len(X), batch_size, replace=False)

        return X[a]
        # return X

    def train(self, data):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epochs):
                # if i % 5 == 0 and self.learning_rate >= 0.0001:
                    # self.learning_rate = self.learning_rate / 2
                    # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                embedding = None
                for j in range(np.shape(data)[0] // self.batch_size):
                    batch_data = self.get_batch(data, self.batch_size)
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data})
                    if embedding is None:
                        embedding = self.encoded
                    else:
                        embedding = np.vstack((embedding, self.encoded))
                    print('batch {0}: loss = {1}'.format(j, l))
                self.embedding = embedding
                # if i % 50 == 0:
                #     print('epoch {0}: loss = {1}'.format(i, l))
                #     self.saver.save(sess, './model/ckpt')
                print('epoch {0}: loss = {1}'.format(i, l))
            self.saver.save(sess, './model.ckpt')
    
    def get_embedding(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './model.ckpt')
            look_back = self.g.look_back_list
            vectors = {}
            embeddings, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: self.g.adjMat})
            for i, embedding in enumerate(embeddings):
                vectors[look_back[i]] = embedding

        return vectors, reconstructed

    def save_embeddings(self, filename):
        self.vectors, reconstructed = self.get_embedding()
        fout = open(filename, 'w')
        fout.write("{} {}\n".format(self.g.node_size, self.hidden_dim))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(str(node),
                                        ' '.join([str(x) for x in vec])))
        
        fout.close()

        # fout1 = open('admat.txt', 'w')
        # for vec in self.g.adjMat:
        #     fout1.write('{}\n'.format(' '.join([str(x) for x in vec])))
        # fout1.close()

        # fout2 = open('reconstructed.txt', 'w')
        # for vec in reconstructed:
        #     fout2.write('{}\n'.format(' '.join([str(x) for x in vec])))
        # fout2.close()