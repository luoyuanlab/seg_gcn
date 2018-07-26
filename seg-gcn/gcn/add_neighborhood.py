
adj_in = np.empty(shape=[0, 0])
for current_ind in range(0,label.shape[0]):
    matched_word_ind = [j for j, item in enumerate(all_size[:,0]) if item in current_ind]
    adj_sing = scipy.sparse.coo_matrix(adj.tocsr()[matched_word_ind, :].tocsc()[:, matched_word_ind])
    
    adj_sing.setdiag(1,k=-1)
    adj_sing.setdiag(1,k=1)
    
    adj_in = scipy.sparse.block_diag((adj_in,adj_sing))