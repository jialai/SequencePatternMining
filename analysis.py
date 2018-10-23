import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("mobsos_MONITORING.csv")
dataOfInvocation=pd.read_csv("MOBSOS_INVOCATION.csv")

#将分类数据转化成序列形式MNAME
#还可以直接转换回去，这个之后在想
class_mapping={label:idx for idx,label in enumerate(np.unique(dataOfInvocation['MNAME']))}
dataOfInvocation['MNAME']=dataOfInvocation['MNAME'].map(class_mapping)
dataOfInvocationOneD=dataOfInvocation[['MNAME']]
print(dataOfInvocationOneD)
dataOfInvocationOneD.to_csv("MobSOSInvocationWithLabelOneD.csv",index=False,sep=',')
#dataOfInvocation.to_csv("MobSOSInvocationWithLabel.csv",index=False,sep=',')

