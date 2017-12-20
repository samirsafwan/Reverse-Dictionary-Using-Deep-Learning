import numpy as np

def attention(input_a, input_b):
    """
    Given input_a = LSTM(input1) and input_b = LSTM(input2),
    returns a tuple of attention vectors, a_tilde and b_tilde
    """
    a = np.array(input_a)
    b = np.array(input_b)
    e = np.zeros((a.shape[0],b.shape[0]))
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            e[i,j] = np.dot(a[i,:],b[j,:])
    
    sumrow = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        sumrow[i] = np.sum([np.exp(e[i,j]) for j in range(b.shape[0])])

    a_tilde = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        a_tilde[i] = np.sum([(np.exp(e[i,j])/sumrow[i])*b[j,:] for j in range(b.shape[0])])

    sumcol = np.zeros(b.shape[0])
    for j in range(b.shape[0]):
        sumcol[j] = np.sum([np.exp(e[i,j]) for i in range(a.shape[0])])

    b_tilde = np.zeros(b.shape[0])
    for j in range(b.shape[0]):
        b_tilde[j] = np.sum([(np.exp(e[i,j])/sumcol[j])*a[i,:] for i in range(a.shape[0])])
    
    return a_tilde, b_tilde