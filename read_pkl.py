import pickle

path = '/home/zy/zyq/grape/result.pkl'
f = open(path,'rb')
data = pickle.load(f)

f2 = open('/home/zy/zyq/grape/result.txt','w')
f2.write(str(data))
print(data)


