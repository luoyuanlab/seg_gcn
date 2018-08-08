# java -mx16g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9003 -timeout 999999


import gensim
import numpy as np
import scipy as sp
import re
import glob
import scipy
from bllipparser import RerankingParser
import StanfordDependencies
import warnings


from collections import defaultdict
from pprint import pprint
# finds shortest path between 2 nodes of a graph using BFS
def bfs_shortest_path(graph, start, goal):
    # keep track of explored nodes
    explored = []
    # keep track of all the paths to be checked
    queue = [[start]]
 
    # return path if start is goal
    if start == goal:
        return []
 
    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in explored:
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path
 
            # mark node as explored
            explored.append(node)
 
    # in case there's no path between the 2 nodes
    return []

warnings.filterwarnings('ignore')

rel_summary_all_doc = np.load("/home/yld8809/all_rel/tp_all_test.npy")
raw_doc_folder = "/home/yld8809/all_rel/txt_all_test/"
max_size = np.load('/home/yld8809/max_size_tp.npy')

model = gensim.models.KeyedVectors.load_word2vec_format("/home/yld8809/semrel/mimic3_pp300.txt", binary=False)  
model_size = 300

rrp = RerankingParser.fetch_and_load('GENIA+PubMed', verbose=True)

raw_ind = glob.glob(raw_doc_folder+'/*.txt')
raw_ind.sort()

num_doc = len(raw_ind)

word_embedding_all = np.empty(shape=[0, model_size+7])
dep_mat_all = np.empty(shape=[0, 0])
de_parse_last = []
last_sent = []

features_tuple = []
adj_tuple = []
sentence_length_tuple = []

count = -1
for i in range(349,349+num_doc):
    count = count + 1
    raw_doc_dir = raw_ind[count] # raw doc 
    
    text_curr = open(raw_doc_dir,encoding='utf-8').read()
    text_curr = text_curr.split('\n')
    index = np.asarray(np.where([int(rel_summary_all_doc[k,2]) == i for k in range(0,len(rel_summary_all_doc[:,2]))])).reshape(-1,1)
    
    for j in index:
        # word embedding
        print(i,j)
        sentence = text_curr[int(np.asscalar(rel_summary_all_doc[j,3]))-1]
#==============================================================================
#         sentence = re.sub('[.]', '', sentence)
#==============================================================================
#==============================================================================
#         sentence = re.sub('[.]', ' , ', sentence)
#         sentence = re.sub('[.]', ' , ', sentence)
#         sentence = re.sub("[)]", " ) ", sentence)
#==============================================================================
        sentence = re.sub(r'p[.]o[.]','post',sentence)
        sentence = re.sub(r't[.]i[.]d[.]','tid',sentence)
        sentence = re.sub(r'\?','question',sentence)
#==============================================================================
#         date_reg_exp_1 = re.compile('\d{2}-year-old')
#         matches_list = date_reg_exp_1.findall(sentence)
#         if matches_list != []:
#         sentence = re.sub(r'[0-9]{2}-year-old',matches_list[0][0:2],sentence)
#==============================================================================
        
         
        
#==============================================================================
#         sentence = re.sub(r'[%]','',sentence)    
#         sentence = re.sub(r'[%]','',sentence)
#         sentence = re.sub(r't[.]i[.]d[.]','tid',sentence)
#         sentence = re.sub(r"'s[)][.]",')',sentence)
#         sentence = re.sub(r'breath\.(.*?)Had','breath had',sentence)
#         sentence = re.sub(r'community-acquired','community',sentence)
#         sentence = re.sub('w/', 'with', sentence)
#==============================================================================
        
#==============================================================================
#         sentence = re.sub('[0-9]{2}[-,:][0-9]{2}', 'index', sentence)
#==============================================================================

        if sentence == last_sent:
            de_parse = de_parse_last
        else:
            # dependency parsing
            sentence_word_m = sentence.split()         
            de_parse = rrp.parse(sentence_word_m)   
            


### Only use lower case
        word_embedding = np.empty(shape=[0, model_size])
        for k in sentence_word_m:
#             if k in model.vocab:
#                 word_embedding = np.vstack((word_embedding,model[k]))
            if k.lower() in model.vocab:
                #word_embedding = np.vstack((word_embedding,model[k.lower()]))
                word_embedding = np.vstack((word_embedding,model[k.lower()]))             
            else:
                word_embedding = np.vstack((word_embedding,np.zeros(shape=[1, model_size])))

#==============================================================================
#         word_embedding = [model[re.sub("[']", '', k.lower())] for k in sentence_word]
#==============================================================================

        word_embedding = np.asarray(word_embedding)
        features_temp = np.zeros(shape=[max_size,model_size])
        features_temp[0:np.asarray(word_embedding).shape[0],:] = word_embedding[:,:]
        features_tuple = features_tuple + [scipy.sparse.coo_matrix(features_temp)]
        
        
        sentence_length = np.asarray(sentence_word_m).shape[0]
        dep_mat = np.zeros(shape=[sentence_length,sentence_length])
        
        if len(de_parse) != 0:  
            tokens = de_parse[0].ptb_parse.sd_tokens()
            
            for token in tokens:
                current_dep = list(token)

                if current_dep[7] != 'root':
                    dep_mat[(current_dep[0]-1),(current_dep[6]-1)] = 1
                    dep_mat[(current_dep[6]-1),(current_dep[0]-1)] = 1
                    
        graph = defaultdict(list)



        for i_s, v_s in enumerate(dep_mat, 1):
            for j_s, u_s in enumerate(v_s, 1):             
                if u_s != 0 and (j_s-1) not in graph[i_s-1]:
                    graph[i_s-1].append(j_s-1)                         
        #
        length_all = np.empty(shape=[0,1])
        all_path = []
        for f_con in range(int(np.asscalar(rel_summary_all_doc[j,4])),int(np.asscalar(rel_summary_all_doc[j,5]))+1):
            for s_con in range(int(np.asscalar(rel_summary_all_doc[j,6])),int(np.asscalar(rel_summary_all_doc[j,7]))+1):
                shortest_path = bfs_shortest_path(graph, f_con, s_con)

                length_all = np.vstack((length_all,len(shortest_path)))
                all_path = all_path + [shortest_path]

        shortest_of_all = all_path[np.where(length_all == min(length_all)[0])[0][0]]
        
        graph = defaultdict(list)

        for i_s, v_s in enumerate(dep_mat, 1):
            for j_s, u_s in enumerate(v_s, 1):                  
                if u_s != 0 and (j_s-1) not in graph[i_s-1]:
                    graph[i_s-1].append(j_s-1)    
        dict_short_path_only = dict((k, graph[k]) for k in shortest_of_all)
    
#         if sorted(shortest_of_all) != []:       
#             for i_s in range(0,sentence_length):
#                 if i_s in (i_s for i_s in range(max(0,min((shortest_of_all)[0],(shortest_of_all)[-1])-(5-1)), 
#                                                     min((shortest_of_all)[0],(shortest_of_all)[-1])+1)):
#                     dict_short_path_only.update({i_s:[]})
#                 elif i_s in (i_s for i_s in range(max((shortest_of_all)[0],(shortest_of_all)[-1]), 
#                                                     min(sentence_length,max((shortest_of_all)[0],(shortest_of_all)[-1])+5))):
#                     dict_short_path_only.update({i_s:[]})

#         for i_s in dict_short_path_only:
#             for j_s in range(0,sentence_length):
#                     if j_s == (i_s-1):
#                         if j_s not in dict_short_path_only[i_s]:
#                             dict_short_path_only[i_s].append(j_s) 
#                     if j_s == (i_s+1):
#                         if j_s not in dict_short_path_only[i_s]:
#                             dict_short_path_only[i_s].append(j_s) 

#                     if i_s in (i_s for i_s in range(max((shortest_of_all)[0],(shortest_of_all)[-1]), 
#                                                     min(sentence_length,max((shortest_of_all)[0],(shortest_of_all)[-1])+5))):
#                         if j_s == (i_s+1):
#                             if j_s not in dict_short_path_only[i_s]:
#                                 dict_short_path_only[i_s].append(j_s)  
#                         if j_s == (i_s+2):
#                             if j_s not in dict_short_path_only[i_s]:
#                                 dict_short_path_only[i_s].append(j_s)                             
#                     else:
#                         if j_s == (i_s+1):
#                             if j_s not in dict_short_path_only[i_s]:
#                                 dict_short_path_only[i_s].append(j_s)  
#                         if j_s == (i_s-1):
#                             if j_s not in dict_short_path_only[i_s]:
#                                 dict_short_path_only[i_s].append(j_s) 
                            
        dep_mat = np.zeros(shape=[max_size,max_size])
#         dep_mat = np.zeros(shape=[sentence_length,sentence_length])
    
        for token in dict_short_path_only:
            for token_n in dict_short_path_only[token]:
                dep_mat[(token),(token_n)] = 1
                dep_mat[(token_n),(token)] = 1
                
        sentence_length_tuple = sentence_length_tuple + [np.asarray(sentence_word_m).shape[0]]    
        adj_tuple = adj_tuple + [scipy.sparse.coo_matrix(dep_mat)]
        
        de_parse_last = de_parse
        last_sent = sentence 
        
        
np.save("/home/yld8809/tp_features_padded_test", features_tuple)
np.save("/home/yld8809/tp_adj_padded_test", adj_tuple)
np.save("/home/yld8809/sentence_length_tuple_test_tp", sentence_length_tuple)