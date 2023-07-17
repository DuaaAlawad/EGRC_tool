seed_value= 0
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import datetime
import tensorflow as tf
tf.random.set_seed(seed_value)
import torch
import sys, copy, math, time, pdb
import pickle as cPickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import argparse
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from inspect import signature
from tqdm import tqdm
import os, sys
import _pickle as cp
import networkx as nx
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pathlib
from sklearn.metrics import classification_report
from tf_geometric.layers import SAGPool, GCN, SortPool,DiffPool, GCN,MinCutPool
from tf_geometric.utils import tf_utils
from numpy import savetxt
import torch.nn.functional as Fs
from tf_geometric.utils.graph_utils import convert_x_to_3d
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tf_geometric as tfg
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
#========================================================================================================================================================
inputfile='/home/dmalawad/Research/EGRC_v2/input.txt'
# Read from the file 
with open(inputfile) as f:
    first_line = f.readline()
    second_line = f.readline()
    third_line = f.readline() 
 # create the Probab files as CSV files
first_line = first_line.strip('\n')
second_line = second_line.strip('\n')
third_line = third_line.strip('\n')
#----------------------------------------------------------------------------------------
sys.path.append(os.path.abspath(first_line))
import torch
device = torch.device('cuda')
import pandas as pd
sys.path.append(os.path.abspath('/home/dmalawad/Research/EGRC_v2/software/node2vec/src'))
from node2vec import *

sys.path.append(os.path.abspath(first_line))
from util_functions import *
sys.argv=['']
np.random.seed(314)
tf.compat.v1.set_random_seed(314)
random.seed(314)
np.set_printoptions(threshold=np.inf)

#=================
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#============================================================== Functions==============================================================================================

def genenet_attribute(allx,tfNum):
    #1: average to one dimension
    allx_ = StandardScaler().fit_transform(allx)
    trainAttributes = np.average(allx_, axis=1).reshape((len(allx),1))
    #2: std,min,max as the attribute
    meanAtt = np.average(allx,axis=1).reshape((len(allx_),1))
    stdAtt = np.std(allx,axis=1).reshape((len(allx_),1))
    minVal = np.min(allx,axis=1).reshape((len(allx_),1))
    # expAtt = allx[:,:536]

    #2 folder
    # qu1Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    # maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    # qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    # qu2Att = (maxVal-qu1Val)/(maxVal-minVal)

    # quantilPerAtt =np.concatenate([qu1Att,qu2Att],axis=1)
    # quantilValAtt =np.concatenate([minVal, qu1Val, maxVal],axis=1)

    #4 folder
    qu1Val = np.quantile(allx,0.25, axis=1).reshape((len(allx_),1))
    qu2Val = np.quantile(allx,0.5, axis=1).reshape((len(allx_),1))
    qu3Val = np.quantile(allx,0.75, axis=1).reshape((len(allx_),1))
    maxVal = np.max(allx,axis=1).reshape((len(allx_),1))

    qu1Att = (qu1Val-minVal)/(maxVal-minVal)
    qu2Att = (qu2Val-qu1Val)/(maxVal-minVal)
    qu3Att = (qu3Val-qu2Val)/(maxVal-minVal)
    qu4Att = (maxVal-qu3Val)/(maxVal-minVal)

    quantilPerAtt =np.concatenate([qu1Att,qu2Att,qu3Att,qu4Att],axis=1)
    quantilValAtt =np.concatenate([minVal, qu1Val,qu2Val,qu3Val,maxVal],axis=1)

    #5: TF or not, vital
    tfAttr = np.zeros((len(allx),1))
    for i in np.arange(tfNum) :
        tfAttr[i]=1.0
    
    #2. PCA to 3 dimensions
    allx_ = StandardScaler().fit_transform(allx)
    pca = PCA(n_components=3)
    pcaAttr = pca.fit_transform(allx_)

    # trainAttributes = np.concatenate([trainAttributes, stdAtt, minAtt, qu1Att, qu3Att, maxAtt, tfAttr], axis=1)
    #trainAttributes = np.concatenate([trainAttributes, stdAtt, tfAttr], axis=1)
    # Describe the slope
    # Best now:
    # trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, tfAttr], axis=1)

    trainAttributes = np.concatenate([trainAttributes], axis=1)
    # trainAttributes = np.concatenate([trainAttributes, stdAtt, quantilPerAtt, quantilValAtt, tfAttr], axis=1)
    
    #trainAttributes = np.concatenate([tfAttr], axis=1)
    
    return trainAttributes



def sample_neg_TF(net, test_ratio=0.1, TF_num=333, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    #TODO
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    recordDict={}
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, TF_num), random.randint(0, n-1)
        if i < j and net[i, j] == 0 and str(i)+"_"+str(j) not in recordDict:
            neg[0].append(i)
            neg[1].append(j)
            recordDict[str(i)+"_"+str(j)]=''
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg



def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1 # inject negative train
        A[col, row] = 1 # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1,workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings





## Original
# Extract subgraph from links 
def extractLinks2subgraphs(Atrain, Atest, train_pos, train_neg, test_pos, test_neg, h=1, max_nodes_per_hop=None, train_node_information=None, test_node_information=None,sk="SP"):
     # automatically select h from {1, 2}
    if h == 'auto': # TODO
        # split train into val_train and val_test
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('\033[91mValidation AUC of AA is {}, CN is {}\033[0m'.format(val_auc_AA, val_auc_CN))
        if val_auc_AA >= val_auc_CN:
            h = 2
            print('\033[91mChoose h=2\033[0m')
        else:
            h = 1
            print('\033[91mChoose h=1\033[0m')
    
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label, node_information,nam):
        g_list = []
        graph_lists=[]
        duaa=0
        for i, j in tqdm(zip(links[0], links[1])):
            # print(" i and j ",i,j,file=Output_file_2)
            g, n_labels, n_features = subgraph_extraction_labeling((i, j), A, h, max_nodes_per_hop, node_information)
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
            if nam=='train_pos' or nam=='train_neg':
                if sk=='MI':
                    print(" iandj",',',i,',',j,',',nam,',',sk,file=Output_train_SP)
                # print(g_label,file=Output_file)

                if sk=='SP':
                    print("iandj",',',i,',',j,',',nam,',',sk,file=Output_train_MI)
            
            
            if nam=='test_pos' or nam=='test_neg':
                if sk=='MI':
                    print(" iandj",',',i,',',j,',',nam,',',sk,file=Output_test_SP)
                # print(g_label,file=Output_file)

                if sk=='SP':
                    print("iandj",',',i,',',j,',',nam,',',sk,file=Output_test_MI)
            duaa=duaa+1
            g_num=duaa
            print("# of graph",duaa,nam,sk,file=Output_file)
            print("***********************",file=Output_file)
            
            # if sk=='MI':
            #     print(" i and j ",',',i,',',j,',',nam,',',sk,file=Output_file_3)
            # print("g ",g)
            degr=np.array(list(dict(g.degree).values()))
            du =duaaGraph(g, g_label, n_labels, n_features,g_num)
            a=du.get_num_nodes()
            b=du.get_num_edges()
            c=du.get_num_node_label()
            aa=du.get_g_num()
            d=du.get_graph_label()
            e=du.get_edge_pairs()
            feat=du.get_node_features()
            # print("=================",feat)
            # Initialize the dictionary
            graphs = {'num_nodes':a,'num_edges':b, 'node_labels':c, 'graph_label':d,'degrees':degr,'edge_index':e,'node_features':feat ,'g_ID':g_num}

            # Create blank list to append to
            graph_lists.append(graphs)
            # print(graph_lists)
            g_list.append(duaaGraph(g, g_label, n_labels, n_features,g_num))
            # print("================")
        return graph_lists
    print('Extract enclosed subgraph...')
    train_g =helper(Atrain, train_pos, 1, train_node_information,nam="train_pos") + helper(Atrain, train_neg, 0, train_node_information,nam="train_neg")
    test_g =helper(Atrain, test_pos, 1, train_node_information,nam="test_pos") + helper(Atrain, test_neg, 0, train_node_information,nam="test_neg")
    print("==============================================================================================================")
    print(max_n_label)
    # return train_graphs, test_graphs, max_n_label['value'],train_g,test_g
    return max_n_label['value'] ,train_g, test_g



def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None, node_information=None):
    print("the main TF=>G  represnts int ",ind,file=Output_file)
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[0], ind[1]])
    print("nodes0",nodes,file=Output_file)
    visited = set([ind[0], ind[1]])
    print("visited0",visited,file=Output_file)
    fringe = set([ind[0], ind[1]])
    print("fringe0",fringe,file=Output_file)
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        print("fringe atfter N",fringe,file=Output_file)
        fringe = fringe - visited
        print("fringe sub Visi N",fringe,file=Output_file)
        visited = visited.union(fringe)
        print("Final visit",visited,file=Output_file)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        print("all subgrapg parts such as main reg (TF=>g) wit its neighbors ",nodes,file=Output_file)
        print("dist",dist,file=Output_file)
        nodes_dist += [dist] * len(fringe)
        print("* represnt 0 for TF and  1 for G",nodes_dist,file=Output_file)
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    print("neighbors nodes ",nodes ,file=Output_file)
    nodes = [ind[0], ind[1]] + list(nodes) 
    print(" list contain",nodes,file=Output_file)
    subgraph = A[nodes, :][:, nodes]
    print("subgraph",subgraph,file=Output_file)
    # print("\n subgraph before ",subgraph,file=Output_file)
    # print("******************",file=Output_file)
    # apply node-labeling
    labels = node_label(subgraph)
    print("node labels",labels,file=Output_file)
    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]
    # construct nx graph
    # print("Sub",subgraph,file=Output_file)
    g = nx.from_scipy_sparse_matrix(subgraph)
    print(g.nodes(),file=Output_file)
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
        print("after removing links between targegt", g,file=Output_file)
    Output_file.flush()
    return g, labels.tolist(), features





# original version
def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels



class duaaGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None,g_num=0):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
       '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.g_num=g_num
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
        # self.data = Data

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
            edge_index = torch.tensor(self.edge_pairs, dtype=torch.long)
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
            edge_index = torch.tensor(self.edge_pairs, dtype=torch.long)
            # self.data = Data(edge_index=edge_index.t().contiguous())

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(edge_features.values()[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

    def __repr__(self):
        return "This is object of class A"

    def get_num_nodes(self):
        return  np.array(self.num_nodes)

    def get_num_edges(self):
        return (self.num_edges)

    def get_num_node_label(self):
        return (self.node_tags)

    def get_graph_label(self):
        return np.array((self.label))
    
    def get_g_num(self):
        return np.array((self.g_num))

    def get_node_features(self):
        return np.array(self.node_features)

    def get_degress(self):
        return np.array(self.degs)

    def get_edge_pairs(self):
      List=self.edge_pairs
      B, C = List[::2], List[1::2]
      mains=[B, C]
      return np.array(mains)

    def get_edge_features(self):
        return np.array(self.edge_features)


    def printinfo(self):
        print("num_nodes",self.num_nodes)
        print("num_edges =", self.num_edges)
        print("node_tags =", self.node_tags)
        print("label =", self.label)
        print("label =", self.g_num)
        print("node_features =", self.node_features)
        print("degs =", self.degs)
        print("edge_pairs =", self.edge_pairs)
        print("edge_features =", self.edge_features)
        print("var2 =", end=" ")    
   

def DGCNN_classifer(train_graphs,test_graphs,epoch_num,p):
    
    t = time.time()
    def create_graph_generator(graphs, batch_size, infinite=False, shuffle=False):
        while True:
            dataset = tf.data.Dataset.range(len(graphs))
        #   if shuffle:
        #       dataset = dataset.shuffle(2000)
            dataset = dataset.batch(batch_size)

            for batch_graph_index in dataset:
                batch_graph_list = [graphs[i] for i in batch_graph_index]

                batch_graph = tfg.BatchGraph.from_graphs(batch_graph_list)
                yield batch_graph

            if not infinite:
                break
    batch_size = 500

    # Multi-layer GCN Model
    class GCNModel(tf.keras.Model):

        def __init__(self, units_list, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.gcns = [
                # tfg.layers.GCN(units, activation=tf.nn.relu if i < len(units_list) - 1 else None)
                tfg.layers.MeanGraphSage(units, concat=False, activation=tf.nn.relu if i < len(units_list) - 1 else None)
                for i, units in enumerate(units_list)
            ]

        def call(self, inputs, training=None, mask=None):
            x, edge_index, edge_weight = inputs
            h = x
            for gcn in self.gcns:
                h = gcn([h, edge_index, edge_weight], training=training)
            return h
    
    
    
    class DiffPoolModel(tf.keras.Model):

        def __init__(self, num_clusters_list, num_features_list, num_classes, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.diff_pools = []

            for num_features, num_clusters in zip(num_features_list, num_clusters_list):
                diff_pool = DiffPool(
                    feature_gnn=GCNModel([num_features, num_features]),
                    assign_gnn=GCNModel([num_features, num_clusters]),
                    units=num_features, num_clusters=num_clusters, activation=tf.nn.relu
                )
                self.diff_pools.append(diff_pool)

            self.mlp = tf.keras.Sequential([
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)
            ])

        def call(self, inputs, training=None, mask=None):
            x, edge_index, edge_weight, node_graph_index = inputs
            h = x
            graph_h_list = []
            for diff_pool in self.diff_pools:
                h, edge_index, edge_weight, node_graph_index = diff_pool([h, edge_index, edge_weight, node_graph_index],
                                                                        training=training)
                graph_h = tfg.nn.max_pool(h, node_graph_index)
                graph_h_list.append(graph_h)

            graph_h = tf.concat(graph_h_list, axis=-1)
            logits = self.mlp(graph_h, training=training)

            return logits

     
    num_clusters_list = [20, 5]
    num_features_list = [128, 128]

    model = DiffPoolModel(num_clusters_list, num_features_list, num_classes) 

    def evaluate():
        accuracy_m = tf.keras.metrics.Accuracy()
        for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
            logits = forward(test_batch_graph)
            preds = tf.argmax(logits, axis=-1)
            accuracy_m.update_state(test_batch_graph.y, preds)
        return accuracy_m.result().numpy()



    def forward(batch_graph, training=False):
      return model([batch_graph.x, batch_graph.edge_index, batch_graph.edge_weight, batch_graph.node_graph_index],
                  training=training)

    def  Calculate_Prob():
            AUPR = tf.keras.metrics.AUC(curve="PR",summation_method='interpolation')
            # AUPR=tfa.metrics.AUCPrecisionRecall()
            count=0
            for test_batch_graph in create_graph_generator(test_graphs, batch_size, shuffle=False, infinite=False):
                # print( test_batch_graph)
                logits = forward(test_batch_graph)
                # print(logits)
                preds = tf.argmax(logits, axis=-1)
            
                prob=np.max(logits, axis = -1)
                predictions=tf.nn.softmax(logits)

                
                savetxt(pooling_path+'/Epoch-'+str(c)+'/_predictions'+p+"_B"+str(count)+'.csv', predictions, delimiter=',')
                savetxt(pooling_path+'/'+'y'+str(count)+'.csv', test_batch_graph.y, delimiter=',')
                count=count+1
                AUPR.update_state(test_batch_graph.y, preds.numpy())
            return AUPR.result().numpy(),count
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    train_batch_generator = create_graph_generator(train_graphs, batch_size, shuffle=True, infinite=True)

    sum_aupr=0
    c=0
    for step in tqdm(range(epoch_num)):
        train_batch_graph = next(train_batch_generator)
        with tf.GradientTape() as tape:
            logits = forward(train_batch_graph, training=True)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=tf.one_hot(train_batch_graph.y, depth=num_classes)
            )

        vars = tape.watched_variables()
        grads = tape.gradient(losses, vars)
        optimizer.apply_gradients(zip(grads, vars))
        
        if step % 20 == 0:
            
            # print("c",c)
            mean_loss = tf.reduce_mean(losses)
            AUPR1,count= Calculate_Prob()
            accuracy = evaluate()
            sum_aupr=sum_aupr+AUPR1
            c=c+1
            print("step = {}\tloss = {}".format(step, mean_loss))
    len_sample=(epoch_num/20)

    Avg_AUPR=sum_aupr/len_sample

    # print("Avg_AUPR",Avg_AUPR)
    
    return accuracy,c,count


    



# #=======================================================Main Code =======================================================================================
from multiprocessing import Process
import sys

rocket = 0

if __name__ == "__main__":
     
   
    time1=datetime.datetime.now()    
    dreamTFdict={}
    dreamTFdict['data1']=195
    dreamTFdict['data3']=334
    dreamTFdict['data4']=333
    dreamTFdict['Human']=745
    dreamTFdict['data5']=30
    dreamTFdict['data6']=30
    dreamTFdict['data7']=159
   
    #parameters
    cuda=True
    embedding_dim=1
    feature_num=4
    hop=1 
    use_embedding=True
    use_attribute='store_true'
    training_ratio=1.0
    max_nodes_per_hop=None
    max_train_num=100000
    mutual_net=3
    neighbors_ratio=1.0 
    no_cuda=False 
    nonezerolabel_flag=False 
    nonzerolabel_ratio=1.0
    pearson_net=0.8
    file_dir=first_line.strip()
   
    dreamTFdict={}
    dreamTFdict['data1']=195
    dreamTFdict['data3']=334
    dreamTFdict['data4']=333
    dreamTFdict['Human']=745
    seed=43
    zerolabel_ratio=0.0
    # file_dir=third_line
    trdata_name=second_line
    tedata_name=third_line
    traindata_name=second_line
    testdata_name=third_line
    epoch_num=100
  
#    #=========================================================== Writing files============================================
  
    # create the Probab files as CSV files
    first_line = first_line.strip('\n')
    second_line = second_line.strip('\n')
    third_line = third_line.strip('\n')
    pooling_path1 =first_line+'/Output/DiffPooling/'+third_line+'/'
    probab_path1 = os.path.join(pooling_path1)
    probab_path1=probab_path1.strip()
    if os.path.exists(probab_path1):
        print("File exist")
    else:
        print ("File not exist")
        os.mkdir(probab_path1)


    # create the Probab files as CSV files
    first_line = first_line.strip('\n')
    pooling_path =pooling_path1+'probab'
    # probab_path = os.path.join(pooling_path, third_line)
    # probab_path=probab_path.strip()
    if os.path.exists(pooling_path):
        print("File exist")
    else:
        print ("File not exist")
        os.mkdir(pooling_path)   
  

        
    namefile_results=pooling_path1+"/results.txt"
    Output_file_results=open(namefile_results,"w")
    time1=datetime.datetime.now()
    
    #Create the output folder 
    namefile=probab_path1+"/ALL_Graphs_DiffPooling.txt"
    Output_file=open(namefile,"w")



    if third_line!='data4':
        path_test_SP=first_line+'/Output/DiffPooling/'+third_line+"/SP_TF_G.txt"
        path_test_MI=first_line+'/Output/DiffPooling/'+third_line+"/MI_TF_G.txt"
        Output_test_SP=open(path_test_SP,"w")
        Output_test_MI=open(path_test_MI,"w")
        print(" i and j ",',','i',',','j',',','nam',',','sk',file=Output_test_SP)
        print(" i and j ",',','i',',','j',',','nam',',','sk',file=Output_test_MI) 
        


        path_train_SP=first_line+'/Output/DiffPooling/'+second_line+"/SP_TF_G.txt"
        path_train_MI=first_line+'/Output/DiffPooling/'+second_line+"/MI_TF_G.txt"
        Output_train_SP=open(path_train_SP,"w")
        Output_train_MI=open(path_train_MI,"w")
        print(" i and j ",',','i',',','j',',','nam',',','sk',file=Output_train_SP)
        print(" i and j ",',','i',',','j',',','nam',',','sk',file=Output_train_MI) 
    #==================================================================Parameters=================================================================================   
        
        
    #======================================================================================================================================================================
        # Select data name
        trdata_name = traindata_name.split('_')[0]
        tedata_name = testdata_name.split('_')[0]

        # Prepare Training
        trainNet_ori = np.load(os.path.join(file_dir, 'data/dataset/ind.{}.csc'.format(traindata_name)),allow_pickle=True)
        trainGroup = np.load(os.path.join(file_dir, 'data/dataset/ind.{}.allx'.format(trdata_name)),allow_pickle=True)
        # Pearson's correlation/Mutual Information as the starting skeletons
        trainNet_agent0 = np.load(file_dir+'/data/dataset/'+trdata_name+'_pmatrix_'+str(pearson_net)+'.npy',allow_pickle=True).tolist()
        trainNet_agent1 = np.load(file_dir+'/data/dataset/'+trdata_name+'_mmatrix_'+str(mutual_net)+'.npy',allow_pickle=True).tolist()
        # Random network as the starting skeletons
        # trainNet_agent0 = np.load(file_dir+'/data/dream/'+trdata_name+'_rmatrix_0.003.npy',allow_pickle=True).tolist()
        # trainNet_agent1 = np.load(file_dir+'/data/dream/'+trdata_name+'_rmatrix_0.003.npy',allow_pickle=True).tolist()

        allx =trainGroup.toarray().astype('float32')
        trainAttributes = genenet_attribute(allx,dreamTFdict[trdata_name])
        # Debug: choose appropriate features in debug
        # trainAttributes = genenet_attribute_feature(allx,dreamTFdict[trdata_name],feature_num)  

        # Prepare Testing
        testNet_ori = np.load(os.path.join(file_dir, 'data/dataset/ind.{}.csc'.format(testdata_name)),allow_pickle=True)
        testGroup = np.load(os.path.join(file_dir, 'data/dataset/ind.{}.allx'.format(tedata_name)),allow_pickle=True)
        # Pearson's correlation/Mutual Information as the starting skeletons
        testNet_agent0 = np.load(file_dir+'/data/dataset/'+tedata_name+'_pmatrix_'+str(pearson_net)+'.npy',allow_pickle=True).tolist()
        testNet_agent1 = np.load(file_dir+'/data/dataset/'+tedata_name+'_mmatrix_'+str(mutual_net)+'.npy',allow_pickle=True).tolist()
        # Random network as the starting skeletons
        # testNet_agent0 = np.load(file_dir+'/data/dream/'+tedata_name+'_rmatrix_0.003.npy',allow_pickle=True).tolist()
        # testNet_agent1 = np.load(file_dir+'/data/dream/'+tedata_name+'_rmatrix_0.003.npy',allow_pickle=True).tolist()

        allxt =testGroup.toarray().astype('float32')
        testAttributes = genenet_attribute(allxt,dreamTFdict[tedata_name])
        # Debug: choose appropriate features in debug
        testAttributes = genenet_attribute_feature(allxt,dreamTFdict[tedata_name],feature_num)

        train_pos, train_neg, _, _ = sample_neg_TF(trainNet_ori, 0.0, TF_num=dreamTFdict[trdata_name], max_train_num=max_train_num)
        use_pos_size = math.floor(len(train_pos[0])*training_ratio)
        use_neg_size = math.floor(len(train_neg[0])*training_ratio)
        train_pos=(train_pos[0][:use_pos_size],train_pos[1][:use_pos_size])
        train_neg=(train_neg[0][:use_neg_size],train_neg[1][:use_neg_size])
        _, _, test_pos, test_neg = sample_neg_TF(testNet_ori, 1.0, TF_num=dreamTFdict[tedata_name], max_train_num=max_train_num)
        # test_pos, test_neg = sample_neg_all_TF(testNet_ori, TF_num=dreamTFdict[tedata_name])


        '''Train and apply classifier'''
        Atrain_agent0 = trainNet_agent0.copy()  # the observed network
        Atrain_agent1 = trainNet_agent1.copy()
        Atest_agent0 = testNet_agent0.copy()  # the observed network
        Atest_agent1 = testNet_agent1.copy()
        Atest_agent0[test_pos[0], test_pos[1]] = 0  # mask test links
        Atest_agent0[test_pos[1], test_pos[0]] = 0  # mask test links
        Atest_agent1[test_pos[0], test_pos[1]] = 0  # mask test links
        Atest_agent1[test_pos[1], test_pos[0]] = 0  # mask test links

        # train_node_information = None
        # test_node_information = None
        if use_embedding:
            train_embeddings_agent0 = generate_node2vec_embeddings(Atrain_agent0, embedding_dim, True, train_neg) #?
            train_node_information_agent0 = train_embeddings_agent0
            test_embeddings_agent0 = generate_node2vec_embeddings(Atest_agent0, embedding_dim, True, test_neg) #?
            test_node_information_agent0 = test_embeddings_agent0

            train_embeddings_agent1 = generate_node2vec_embeddings(Atrain_agent1, embedding_dim, True, train_neg) #?
            train_node_information_agent1 = train_embeddings_agent1
            test_embeddings_agent1 = generate_node2vec_embeddings(Atest_agent1, embedding_dim, True, test_neg) #?
            test_node_information_agent1 = test_embeddings_agent1
        if use_attribute and trainAttributes is not None: 
            if use_embedding:
                train_node_information_agent0 = np.concatenate([train_node_information_agent0, trainAttributes], axis=1)
                test_node_information_agent0 = np.concatenate([test_node_information_agent0, testAttributes], axis=1)

                train_node_information_agent1 = np.concatenate([train_node_information_agent1, trainAttributes], axis=1)
                test_node_information_agent1 = np.concatenate([test_node_information_agent1, testAttributes], axis=1)
            else:
                train_node_information_agent0 = trainAttributes
                test_node_information_agent0 = testAttributes

                train_node_information_agent1 = trainAttributes
                test_node_information_agent1 = testAttributes


        graph_dicts_path=first_line+'/'+second_line+'_SP.pkl'
        graph_test_dicts_path=first_line+'/'+third_line+'_SP.pkl'
        graph_dicts1_path=first_line+'/'+second_line+'_MI.pkl'
        graph_test_dicts1_path=first_line+'/'+third_line+'_MI.pkl'
        
               
        
        # =================================================  Extract Subgraph for training and testing dataset ============================================================
        max_n_label,graph_dicts,graph_test_dicts= extractLinks2subgraphs(Atrain_agent0, Atest_agent0, train_pos, train_neg, test_pos, test_neg, hop, max_nodes_per_hop, train_node_information_agent0, test_node_information_agent0,sk="SP")
        max_n_label1,graph_dicts1,graph_test_dicts1 = extractLinks2subgraphs(Atrain_agent1, Atest_agent1, train_pos, train_neg, test_pos, test_neg, hop, max_nodes_per_hop, train_node_information_agent1, test_node_information_agent1,sk="MI")

        Output_test_SP.close()
        Output_test_MI.close()

        Output_train_SP.close()
        Output_train_MI.close()
        
        # #saving subgraph in pickle files 
        f = open(graph_dicts_path,'wb')
        pickle.dump(graph_dicts,f)
        f.close()

        f = open(graph_dicts1_path,'wb')
        pickle.dump(graph_dicts1,f)
        f.close()

        f = open(graph_test_dicts_path,'wb')
        pickle.dump(graph_test_dicts,f)
        f.close()

        f = open(graph_test_dicts1_path,'wb')
        pickle.dump(graph_test_dicts1,f)
        f.close()
        #===================================================================================
    else:
        
        graph_dicts_path=first_line+'/'+second_line+'_SP.pkl'
        graph_test_dicts_path=first_line+'/'+third_line+'_SP.pkl'
        graph_dicts1_path=first_line+'/'+second_line+'_MI.pkl'
        graph_test_dicts1_path=first_line+'/'+third_line+'_MI.pkl'
        
        
        if os.path.exists(graph_dicts_path):
            # load subgraph from picke files 
            with open(first_line+'/'+second_line+'_SP.pkl', 'rb') as f:
                graph_dicts = pickle.load(f)
        
        if os.path.exists(graph_dicts_path):
            with open(first_line+'/'+third_line+'_SP.pkl', 'rb') as f:
                graph_test_dicts = pickle.load(f)
        
        if os.path.exists(graph_dicts1_path):
            with open(first_line+'/'+second_line+'_MI.pkl', 'rb') as f:
                graph_dicts1 = pickle.load(f)
            
        if os.path.exists(graph_test_dicts1_path):
            with open(first_line+'/'+third_line+'_MI.pkl', 'rb') as f:
                 graph_test_dicts1 = pickle.load(f)

    #_____________________________________________________________________________
    
    print("the size of training dataset ",len(graph_dicts))
    print("the size of testing dataset ",len(graph_test_dicts))

    #CREATE Graph using MI Skeletons
    num_node_labels = np.max([np.max(graph_dict1["node_labels"]) for graph_dict1 in graph_dicts1]) + 1
  
    def convert_node_labels_to_one_hot(node_labels):
        num_nodes = len(node_labels)
        print("num_node_labels",num_node_labels)
        x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
        x[list(range(num_nodes)), node_labels] = 1.0
        return x


    def convert_node_featues(graph_dict1):
        x=graph_dict1["node_features"].tolist()
        return x

    def construct_graph(graph_dict1):
        graph_dict1["graph_label"]=np.array(int(graph_dict1["graph_label"])).ravel()
        return tfg.Graph(x=convert_node_featues(graph_dict1), edge_index=graph_dict1["edge_index"], y=np.array(graph_dict1["graph_label"]) ) 

    graphs_MI = [construct_graph(graph_dict1) for graph_dict1 in graph_dicts1]
    num_classes = np.max([graph.y for graph in graphs_MI]) + 1


    #================================

    #CREATE test graphs using MI Skeletons
    num_node_labels = np.max([np.max(graph_test_dict1["node_labels"]) for graph_test_dict1 in graph_test_dicts1]) + 1
  
    def convert_node_labels_to_one_hot(node_labels):
        num_nodes = len(node_labels)
        # print("num_node_labels",num_node_labels)
        x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
        x[list(range(num_nodes)), node_labels] = 1.0
        return x


    def convert_node_featues(graph_test_dict1):
        x=graph_test_dict1["node_features"].tolist()
        return x

    def construct_graph(graph_test_dict1):
        graph_test_dict1["graph_label"]=np.array(int(graph_test_dict1["graph_label"])).ravel()
        return tfg.Graph(x=convert_node_featues(graph_test_dict1), edge_index=graph_test_dict1["edge_index"], y=np.array(graph_test_dict1["graph_label"]) ) 

    graphs_test_MI = [construct_graph(graph_test_dict1) for graph_test_dict1 in graph_test_dicts1]
    num_test_classes = np.max([graph.y for graph in graphs_test_MI]) + 1

#*************************************************************************************************************************************************************
    #CREATE training graph using PC Skeletons
    num_node_labels = np.max([np.max(graph_dict["node_labels"]) for graph_dict in graph_dicts]) + 1
  
    def convert_node_labels_to_one_hot(node_labels):
        num_nodes = len(node_labels)
        print("num_node_labels",num_node_labels)
        x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
        x[list(range(num_nodes)), node_labels] = 1.0
        return x


    def convert_node_featues(graph_dict):
        x=graph_dict["node_features"].tolist()
        return x

    def construct_graph(graph_dict):
        graph_dict["graph_label"]=np.array(int(graph_dict["graph_label"])).ravel()
        return tfg.Graph(x=convert_node_featues(graph_dict), edge_index=graph_dict["edge_index"], y=np.array(graph_dict["graph_label"]) ) 

    graphs_PC = [construct_graph(graph_dict) for graph_dict in graph_dicts]
    num_classes = np.max([graph.y for graph in graphs_PC]) + 1
#===================================================================================

    #CREATE test graph using PC Skeletons
    num_node_labels = np.max([np.max(graph_test_dict["node_labels"]) for graph_test_dict in graph_test_dicts]) + 1
    
    def convert_node_labels_to_one_hot(node_labels):
        num_nodes = len(node_labels)
        print("num_node_labels",num_node_labels)
        x = np.zeros([num_nodes, num_node_labels], dtype=np.float32)
        x[list(range(num_nodes)), node_labels] = 1.0
        return x


    def convert_node_featues(graph_test_dict):
        x=graph_test_dict["node_features"].tolist()
        return x

    def construct_graph(graph_test_dict):
        graph_test_dict["graph_label"]=np.array(int(graph_test_dict["graph_label"])).ravel()
        return tfg.Graph(x=convert_node_featues(graph_test_dict), edge_index=graph_test_dict["edge_index"], y=np.array(graph_test_dict["graph_label"]) ) 

    graphs_test_PC = [construct_graph(graph_test_dict) for graph_test_dict in graph_test_dicts]
    num_test_classes = np.max([graph.y for graph in graphs_test_PC]) + 1

    print("=============================================================================================")
    #test
    print("number of test subgraph using SP skeleton",len(graphs_test_PC))
    print("number of test subgraph using MI skeleton",len(graphs_test_MI))

    #trining
    print("number of training subgraph using SP skeleton",len(graphs_PC))
    print("number of training subgraph using MI skeleton",len(graphs_MI))
    
    #----------------------------------------------------------------Create Epoch Folders------------------------------------------------------------------
    print("Create Epoch Folders ... ")
    c=int(epoch_num/20)
    path=pooling_path
    for i in range (0,c):
        os.chdir(path)
        Newfolder='Epoch-'+str(i)
        try:
            if not os.path.exists(Newfolder):
                os.makedirs(Newfolder)
        except OSError:
            print('ee')

    
    
#========================================================Running the GCN =======================================================================================
    accuracy,c,count=DGCNN_classifer(graphs_MI,graphs_test_MI,epoch_num,p="MI")
    accuracy,c,count=DGCNN_classifer(graphs_PC,graphs_test_PC,epoch_num,p="SP")
    Output_file.close()
#====================================================== Performance Evaluation  ==================================================================================
#====================================================== Performance evaluation for subgraph prediction based on Mutual information  ====================================================
    for epoch_num in range(0,c):
        row_list_MI=[]
        for cot in range(0,count):
            # print(pooling_path+'/Epoch-'+str(epoch_num)+'/predictionsMI_B'+str(cot)+'.csv')
            data_j_i_MI=pd.read_csv(pooling_path+'/Epoch-'+str(epoch_num)+'/_predictionsMI_B'+str(cot)+'.csv',header=None,index_col=False)
            row_list_MI.append(data_j_i_MI)
        df_MI=pd.concat(row_list_MI,axis=0)
        df_col_list_MI = pd.DataFrame(df_MI)
        df_col_list_MI.to_csv(pooling_path+'/Epoch-'+str(epoch_num)+'/df_MI_'+str(epoch_num)+'.csv', index=False)
            

    all_epoch=[]
    for num_epoch in range (0,c):
        df_MI = pd.read_csv(pooling_path + "/Epoch-"+str(num_epoch)+'/df_MI_'+str(num_epoch)+'.csv',header=None,index_col=False)
        all_epoch.append(df_MI)
    data_MI = pd.concat(all_epoch, axis=1)
    # print((data_MI))
    co=int(c)
    header_count_MI = list(range(co*2))
    data_MI.columns=[header_count_MI]
    data_MI.drop(data_MI.index[0])
    # print(data_MI)
    even_id=data_MI.loc[:, 0:num_epoch:2]
    odd_id=data_MI.loc[:, 1:num_epoch:2]
    even_id['MI_avg_0'] = even_id.mean(axis=1)
    odd_id['MI_avg_1'] = odd_id.mean(axis=1)
    e=even_id.iloc[:,-1:]
    o=odd_id.iloc[:,-1:]
    result_MI = pd.concat([e, o], axis=1, join='inner')
    result_MI=result_MI.drop(0)

    # result=result.reset_index()
    result1=result_MI.reset_index()
    result1.to_csv(probab_path1+"final_MI.csv")
    pred1=result1
    #=========================================
    prediction_MI= pred1[['MI_avg_0','MI_avg_1']]
    prediction_MI.columns =['MI_avg0', 'MI_avg1'] 
  
    #===================================================================================
    MI_file = pd.read_csv (pooling_path1+'/MI_TF_G.txt', None)
    MI_file.to_csv (pooling_path1+'MI_TF_G.csv')
    MI_file.columns =['iandj', 'TF', 'G', 'typ','sub'] 
    MI_file['TF']+=1
    MI_file['G']+=1
    MI_file['TF'] = 'G' + MI_file['TF'].astype(str)
    MI_file['G'] = 'G' + MI_file['G'].astype(str)
    df_MI=pd.concat([prediction_MI, MI_file], axis = 1) 
    # print(df_MI.columns)
    pos=df_MI.loc[df_MI['typ'].str.contains('pos')]
    del pos['MI_avg0']
    pos.rename(columns = {'MI_avg1':'probab'}, inplace = True)
    neg=df_MI.loc[df_MI['typ'].str.contains('neg')]
    del neg['MI_avg1']
    neg.rename(columns = {'MI_avg0':'probab'}, inplace = True)
    final = pd.concat([pos, neg], axis=0,join='inner')
    #-------------------------------
    gold = pd.read_csv (first_line+'/data/'+third_line+'/GoldStandard.tsv', header=None, sep='\t')
    gold.rename(columns = {0:'TF',1:'G',2:'true_y'}, inplace = True)
    final_df_MI = pd.merge(final, gold, on=['TF','G'])
    
    print("===============================  the results for subgraph based Mutual Information ==============================================================")
    final_df_MI.to_csv(pooling_path1+'/final_df_MI.csv')
    print("AUROC = ",roc_auc_score(final_df_MI['true_y'], final_df_MI['probab']))
    print("AUPR = ",average_precision_score(final_df_MI['true_y'], final_df_MI['probab']))
    predictions_class = np.where(final_df_MI['probab'] > 0.5, 1, 0)  
    print('Precision: %.3f' % precision_score(y_true=final_df_MI['true_y'], y_pred=predictions_class ))
    print('Recall: %.3f' % recall_score(y_true=final_df_MI['true_y'], y_pred=predictions_class ))
    print('Accuracy: %.3f' % accuracy_score(y_true=final_df_MI['true_y'], y_pred=predictions_class ))
    print('balanced Accuracy: %.3f' % balanced_accuracy_score(y_true=final_df_MI['true_y'], y_pred=predictions_class ))
    
    
    #============================================== Performance evaluation for subgraph prediction based on Spearmean corrlation   ======================
    #sp
    col_list=[]
    for epoch_num in range(0,c):
        row_list=[]
        for cot in range(0,count):
            # print(pooling_path+'/Epoch-'+str(epoch_num)+'/predictionsSP_B'+str(cot)+'.csv')
            data_j_i=pd.read_csv(pooling_path+'/Epoch-'+str(epoch_num)+'/_predictionsSP_B'+str(cot)+'.csv',header=None,index_col=False)
            row_list.append(data_j_i)
        df=pd.concat(row_list,axis=0)
        df_col_list = pd.DataFrame(df)
        #print out all batchs togther in fold 0 + fold 1 + fold 2
        df_col_list.to_csv(pooling_path+'/Epoch-'+str(epoch_num)+'/df_SP_'+str(epoch_num)+'.csv', index=False)
            

    all_epoch=[]
    for num_epoch in range (0,c):
        dfz = pd.read_csv(pooling_path + "/Epoch-"+str(num_epoch)+'/df_SP_'+str(num_epoch)+'.csv',header=None,index_col=False)
        all_epoch.append(dfz)
    data = pd.concat(all_epoch, axis=1)
    co=int(c)
    tons = list(range(co*2))
    data.columns=[tons]
    data.drop(data.index[0])
    # print(data)
    even=data.loc[:, 0:num_epoch:2]
    odd=data.loc[:, 1:num_epoch:2]
    even['SP_avg_0'] = even.mean(axis=1)
    odd['SP_avg_1'] = odd.mean(axis=1)
    e=even.iloc[:,-1:]
    o=odd.iloc[:,-1:]
    result = pd.concat([e, o], axis=1, join='inner',)
    result=result.drop(0)

    # result=result.reset_index()
    result=result.reset_index()
    result.to_csv(probab_path1+"Predication_SP.csv")
    pred=result

    prediction= pred[['SP_avg_0','SP_avg_1']]
    prediction.columns =['SP_avg0', 'SP_avg1'] 
  
    #===================================================================================
    SP_file = pd.read_csv (pooling_path1+'/SP_TF_G.txt', None)
    SP_file.to_csv (pooling_path1+'SP_TF_G.csv')
    SP_file.columns =['iandj', 'TF', 'G', 'typ','sub'] 
    SP_file['TF']+=1
    SP_file['G']+=1
    SP_file['TF'] = 'G' + SP_file['TF'].astype(str)
    SP_file['G'] = 'G' + SP_file['G'].astype(str)
    df_SP=pd.concat([prediction, SP_file], axis = 1) 
    # print(df_SP.columns)
    pos=df_SP.loc[df_SP['typ'].str.contains('pos')]
    del pos['SP_avg0']
    pos.rename(columns = {'SP_avg1':'probab'}, inplace = True)
    neg=df_SP.loc[df_SP['typ'].str.contains('neg')]
    del neg['SP_avg1']
    neg.rename(columns = {'SP_avg0':'probab'}, inplace = True)
    final = pd.concat([pos, neg], axis=0,join='inner')
    #-------------------------------
    gold = pd.read_csv (first_line+'/data/'+third_line+'/GoldStandard.tsv', header=None, sep='\t')
    gold.rename(columns = {0:'TF',1:'G',2:'true_y'}, inplace = True)
    final_df_SP = pd.merge(final, gold, on=['TF','G'])
    print("======================================= the results for subgraph based Speaman corrlation ===================================================")
    final_df_SP.to_csv(pooling_path1+'/final_df_SP.csv')
    print("AUROC = ",roc_auc_score(final_df_SP['true_y'], final_df_SP['probab']))
    print("AUPR = ",average_precision_score(final_df_SP['true_y'], final_df_SP['probab']))
    predictions_class = np.where(final_df_SP['probab'] > 0.5, 1, 0)  
    print('Precision: %.3f' % precision_score(y_true=final_df_SP['true_y'], y_pred=predictions_class ))
    print('Recall: %.3f' % recall_score(y_true=final_df_SP['true_y'], y_pred=predictions_class ))
    print('Accuracy: %.3f' % accuracy_score(y_true=final_df_SP['true_y'], y_pred=predictions_class ))
    print('balanced Accuracy: %.3f' % balanced_accuracy_score(y_true=final_df_SP['true_y'], y_pred=predictions_class ))
    
    
    
    #===========================================================================Performance evaluation for subgraph prediction based on both subgraphs {MI + SP} =========
    #----------------------------------Ensemble---------------
    frames = [result, result1]
    ALL_A = pd.concat(frames,axis=1, join='inner')

    ALL_A['average_0'] = ALL_A[['SP_avg_0', 'MI_avg_0']].mean(axis=1)
    ALL_A['average_1'] = ALL_A[['SP_avg_1', 'MI_avg_1']].mean(axis=1)

    data= ALL_A[['average_0', 'average_1']]
    data.columns =['avg_0', 'avg_1'] 
    data.to_csv(probab_path1+'/all_probab.csv')
    read_file = pd.read_csv (pooling_path1+'MI_TF_G.txt', None)
    read_file.to_csv (pooling_path1+'MI_TF_G.csv')
    # print("data",data.shape)
    # print("read_file",read_file.shape)

    #===========================================================================================
    read_file.columns =['iandj', 'TF', 'G', 'typ','sub'] 
    read_file['TF']+=1
    read_file['G']+=1
    read_file['TF'] = 'G' + read_file['TF'].astype(str)
    read_file['G'] = 'G' + read_file['G'].astype(str)
    df_3=pd.concat([data, read_file], axis = 1)
    # print(df_3.shape)
    pos=df_3.loc[df_3['typ'].str.contains('pos')]
    del pos['avg_0']
    pos.rename(columns = {'avg_1':'probab'}, inplace = True)

    neg=df_3.loc[df_3['typ'].str.contains('neg')]
    del neg['avg_1']
    neg.rename(columns = {'avg_0':'probab'}, inplace = True)
    
    final = pd.concat([pos, neg], axis=0,join='inner')
    #-------------------------------
    gold = pd.read_csv (first_line+'/data/'+third_line+'/GoldStandard.tsv', header=None, sep='\t')
    gold.rename(columns = {0:'TF',1:'G',2:'true_y'}, inplace = True)
    
    #combine
    df_evaluate = pd.merge(final, gold, on=['TF','G'])
    df_evaluate.to_csv(pooling_path1+'/final.csv')
        
    #===================================RESULTS======================================================

    print("====================================Ensemble two subgraps predications (MI + SP )=================================")
    print("AUROC = ",roc_auc_score(df_evaluate['true_y'], df_evaluate['probab']))
    print("AUPR = ",average_precision_score(df_evaluate['true_y'], df_evaluate['probab']))
    predictions_class = np.where(df_evaluate['probab'] > 0.5, 1, 0)  
    conf_matrix = confusion_matrix(y_true=df_evaluate['true_y'], y_pred=predictions_class)
    #
    # Print the confusion matrix using Matplotlib
    #
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    plt.savefig(first_line+'/figures/'+third_line+'_DiffPool.png')
    
    print('Precision: %.3f' % precision_score(y_true=df_evaluate['true_y'], y_pred=predictions_class ))
    print('Recall: %.3f' % recall_score(y_true=df_evaluate['true_y'], y_pred=predictions_class ))
    print('Accuracy: %.3f' % accuracy_score(y_true=df_evaluate['true_y'], y_pred=predictions_class ))
    print('balanced Accuracy: %.3f' % balanced_accuracy_score(y_true=df_evaluate['true_y'], y_pred=predictions_class ))

    print('F1 Score: %.3f' % f1_score(y_true=df_evaluate['true_y'], y_pred=predictions_class ))
    time2=datetime.datetime.now()
    running_time=time2-time1     
    # print("the running time is",running_time,file=Output_file_results) 
    # print("AUROC = ",roc_auc_score(df_evaluate['true_y'], df_evaluate['probab']))
    # print("AUROC = ",average_precision_score(df_evaluate['true_y'], df_evaluate['probab']))
    print("the running time is",running_time) 






















