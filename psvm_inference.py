from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import secretflow as sf
from sklearn import svm
from sklearn import metrics
import jax.numpy as jnp 
import jax.lax
gamma = 1

#feature = ['LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency']
#set = pd.read_csv(open(r'./CTG.CSV'))
#x_train,x_test,y_train,y_test = train_test_split(
#        set[feature],set['NSP'],test_size=0.05,random_state=1,stratify=set['NSP'])
#model = svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovr')
#model.fit(x_train,y_train)
#validation = model.predict(x_test)
#print('val acc:',metrics.accuracy_score(validation,y_test))
#print(model.dual_coef_)
#C=model.dual_coef_
#C_=[0]*len(C[0])
#for i in range(len(C[0])):
#    C_[i]=C[0][i]+C[1][i]
#print(len(C_))
#print(x_test.shape)
#print(len(y_test))
#B=model.intercept_
#X=model.support_vectors_
#T=x_test
#nv=model.n_support_
#cs=model.classes_
X=list(np.random.rand(2000,21))
B=list(np.random.rand(1,3))
C_=list(np.random.rand(1,2000))
T=list(np.random.rand(1,21))
sf.shutdown()
sf.init(['alice','bob'],address = 'local',num_cpus = 4)
alice,bob=sf.PYU('alice'),sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(parties=['alice','bob']))
print(spu.cluster_def)
device = spu
X_,B_,C_,T_=(sf.to(alice,X).to(device),
        sf.to(alice,B).to(device),
        sf.to(alice,C_).to(device),
        sf.to(alice,T).to(device),
)
def SKFC(X,T):
    result = [0]*len(X)
    for i in range(len(X)):
        result[i] = jnp.exp((-1)*jnp.dot(X[i]-T,X[i]-T))
    return list(result)
    #def loop_skfc(i,stat):
    #   res = stat
    #   res[i]=jnp.exp((-gamma)*jnp.dot(X[i]-T,X[i]-T))
    #   return(res)
    #result = jax.lax.fori_loop(0,1687,loop_skfc,result)
    #return result
    
def SDFC(X,C,B,T):
    k=SKFC(X,T)
    c = [0]*3
    c[0] = jnp.sum(jnp.array(list(jnp.sum(C[p]*k[p]) for p in range(0,1310))))
    c[1] = jnp.sum(jnp.array(list(jnp.sum(C[p]*k[p]) for p in range(1311,1687))))
    c[2] = jnp.sum(jnp.array(list(jnp.sum(C[p]*k[p]) for p in range(1687,2000))))
    for i in range(len(B)):
        c[i] = c[i]+B[i]
    return list(c)
    #return [jnp.sum(x) for x in zip(c,B)]

result_=device(SDFC,num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_COMPILER)(X_,C_,B_,T_)
#result = result_.to(alice)
result = sf.reveal(result_)
print(result.shape)
#result = result-y_test
#result = np.count_nonzero(result)
#acc = result/107
#print('acc:',acc)
