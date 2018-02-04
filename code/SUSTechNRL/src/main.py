import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from lib.graph import *
from lib.classify import Classifier, read_node_label

from lib.grarep import GraRep
from lib import line
from lib import node2vec
from lib.sdne import sdne

import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input',
                        default='data/cora/cora_edgelist.txt',
                        help='input graph file')
    parser.add_argument('--output',
                        default='output/cora_deepwalk_embedding.txt',
                        help='output representation file')
    parser.add_argument('--label-file',
                        default='data/cora/cora_labels.txt',
                        help='the file of node label')
    parser.add_argument('--feature-file',
                        default='',
                        help='the file of node features')
    parser.add_argument('--graph-format',
                        default='edgelist',
                        help='input graph format')
    parser.add_argument('--directed',
                        action='store_true',
                        help='treat graph as directed')
    parser.add_argument('--weighted',
                        action='store_true',
                        help='treat graph as weighted')

    parser.add_argument('--representation-size',
                        default=128,
                        type=int,
                        help='number of latent dimensions to learn for each node')
    parser.add_argument('--clf_ratio',
                        default=0.1,
                        type=float,
                        help='the ratio of training data in the classification')

    parser.add_argument('--method',
                        default='deepwalk',
                        choices=['grarep', 'line', 'deepwalk', 'node2vec', 'sdne'],
                        help='the learning method')

    ## parameters for GraRep
    parser.add_argument('--Kstep',
                        default=4,
                        type=int,
                        help='use k-step transition probability matrix')

    ## parameters for LINE
    parser.add_argument('--order',
                        default=3,
                        type=int,
                        help='choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--epochs',
                        default=5,
                        type=int,
                        help='the training epochs of LINE or GCN')
    parser.add_argument('--no-auto-stop',
                        action='store_true',
                        help='no early stop when training LINE')
    
    ## parameters for deepwalk and node2vec
    parser.add_argument('--walk-length',
                        default=80,
                        type=int,
                        help='length of the random walk')
    parser.add_argument('--number-walks',
                        default=10,
                        type=int,
                        help='number of random walks to start at each node')
    parser.add_argument('--window-size',
                        default=10,
                        type=int,
                        help='window size of skipgram model')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='number of parallel processes')
    parser.add_argument('--p',
                        default=1.0,
                        type=float)
    parser.add_argument('--q',
                        default=1.0,
                        type=float)

    ## parameters for SDNE
    parser.add_argument('--struct',
                        default=[None, 1000, None])
    parser.add_argument('--alpha',
                        default=500)
    parser.add_argument('--gamma',
                        default=1)
    parser.add_argument('--reg',
                        default=1)
    parser.add_argument('--beta',
                        default=10)
    parser.add_argument('--batch-size',
                        default=64)
    parser.add_argument('--epochs-limit',
                        default=1)
    parser.add_argument('--learning-rate',
                        default=0.01)
    parser.add_argument('--display',
                        default=1)
    parser.add_argument('--DBN-init',
                        default=True)
    parser.add_argument('--dbn-epochs',
                        default=20)
    parser.add_argument('--dbn-batch-size',
                        default=64)
    parser.add_argument('--dbn-learning-rate',
                        default=0.1)
    parser.add_argument('--sparse-dot',
                        default=False)
    parser.add_argument('--negative-ratio',
                        default=0,
                        type=int,
                        help='the negative ratio of LINE or SDNE')
    
    args = parser.parse_args()

    return args

def buildGraph(args):
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input,
                        weighted=args.weighted,
                        directed=args.directed)
    
    return g

def buildModel(args, g):
    if args.method == 'grarep':
        model = GraRep(graph=g,
                        Kstep=args.Kstep,
                        dim=args.representation_size)
    elif args.method == 'line':
        if args.label_file and not args.no_auto_stop:
            model = line.LINE(graph=g,
                                epoch=args.epochs,
                                rep_size=args.representation_size,
                                order=args.order,
                                label_file=args.label_file,
                                clf_ratio=args.clf_ratio)
        else:
            model = line.LINE(graph=g,
                                epoch=args.epochs,
                                rep_size=args.representation_size,
                                order=args.order)
    elif args.method == 'deepwalk':
        model = node2vec.Node2vec(graph=g,
                                    path_length=args.walk_length,
                                    num_paths=args.number_walks,
                                    dim=args.representation_size,
                                    workers=args.workers,
                                    window=args.window_size,
                                    dw=True)
    elif args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g,
                                    path_length=args.walk_length,
                                    num_paths=args.number_walks,
                                    dim=args.representation_size,
                                    workers=args.workers,
                                    p=args.p,
                                    q=args.q,
                                    window=args.window_size)
    elif args.method == 'sdne':
        model = sdne.SDNE(graph=g,
                            struct=args.struct,
                            alpha=args.alpha,
                            gamma=args.gamma,
                            reg=args.reg,
                            beta=args.beta,
                            batch_size=args.batch_size,
                            epochs_limit=args.epochs_limit,
                            learning_rate=args.learning_rate,
                            display=args.display,
                            DBN_init=args.DBN_init,
                            dbn_epochs=args.dbn_epochs,
                            dbn_batch_size=args.dbn_batch_size,
                            dbn_learning_rate=args.dbn_learning_rate,
                            sparse_dot=args.sparse_dot,
                            negative_ratio=args.negative_ratio,
                            dim = args.representation_size,
                            output_file=args.output)
    
    return model

def evaluate(args, model):
    vectors = model.vectors
    X, Y = read_node_label(args.label_file)
    print('training classifier using {:.2f}% nodes...'.format(args.clf_ratio*100))
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, args.clf_ratio)

def main(args):
    t1 = time.time()
    g = buildGraph(args)
    model = buildModel(args, g)
    t2 = time.time()

    print('training time: %s' % (t2-t1))

    # if args.method != 'gcn' or args.method != 'sdne':
    if args.method != 'gcn':
        print('saving embeddings...')
        model.save_embeddings(args.output)
    if args.label_file and args.method != 'gcn':
        evaluate(args, model)

if __name__ == '__main__':
    random.seed(128)
    np.random.seed(128)
    main(parse_args())
