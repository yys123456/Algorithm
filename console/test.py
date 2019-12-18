from console import *
import copy
# p=process("dataSVM.xlsx",4,[1,'linear',0])
# p=process("dataSVM.xlsx",4,[1,'poly',0])
# result,plt,plt_Hyper=p.start()
# fq,tree,rules=p.start()
# p.show()
# ph.show()
# print(ret)
# plt.show()
# plt_Hyper.show()
# plt_Hyper.show()
# for i in rules:
# 	print(i,rules[i])
# ret,plt_raw,plt_Hyper=p.start()
# result,conf=p.start()
# print(result,conf)

# print(ret)
# plt_raw.show()
# plt_Hyper.show()
# print(result)
# plt.show()
# for key in relatSet:
# 	print(key,"=>",relatSet[key])
# pd.set_option('display.max_rows', 1000)
# print(classInfo)

# plt.show()



#APRIORI测试
'''
p=process("./datas/test/testAP.xlsx",1,[2,0.6])
freq,rules=p.start()
for i in freq:
	print(i)
for i in rules:
	print(i,rules[i])
'''

#FP-GROWTH测试
'''
p=process("./datas/test/testAP.xlsx",2,[2,0.6])
freq,rules,plt=p.start()
for i in freq:
	print(i)
for i in rules:
	print(i,rules[i])
plt.show()
'''
#kmeans测试
#一维
'''
p=process("./datas/test/testKM1.xlsx",3,[3,3])
cif,plt=p.start()
for i in cif:
	print(i)
plt.show()
'''
#二维
'''
p=process("./datas/test/testKM2.xlsx",3,[3,3])
cif,plt=p.start()
for i in cif:
	print(i)
plt.show()
'''
#kmedoids测试
#一维
'''
p=process("./datas/test/testKC1.xlsx",5,[3])
cif,plt=p.start()
for i in cif:
	print(i)
plt.show()
'''
#二维
'''
p=process("./datas/test/testKC2.xlsx",5,[3])
cif,plt=p.start()
for i in cif:
	print(i)
plt.show()
'''
#SVM测试,使用二维数据
#使用高斯核
'''
p=process("./datas/test/testSVM.xlsx",4,[1,'rbf',1])#惩罚系数1,高斯核函数RBF,需要ROC曲线
ret,raw_plt,hyper_plt,roc_plt=p.start()#返回分类结果，原始类别，超平面图，ROC曲线
for i in ret:
	print(i)
raw_plt.show()
hyper_plt.show()
roc_plt.show()
'''
#使用多项式核
'''
p=process("./datas/test/testSVM.xlsx",4,[1,'poly',1])#惩罚系数1,Ploy,需要ROC曲线
ret,raw_plt,hyper_plt,roc_plt=p.start()#返回分类结果，原始类别，超平面图，ROC曲线
for i in ret:
	print(i)
raw_plt.show()
hyper_plt.show()
roc_plt.show()
'''
#使用Sigmoid核

p=process("./datas/test/testSVM.xlsx",4,[1,'sigmoid',1])#惩罚系数1,sigmoid,需要ROC曲线
ret,raw_plt,hyper_plt,roc_plt=p.start()#返回分类结果，原始类别，超平面图，ROC曲线
for i in ret:
	print(i)
raw_plt.show()
hyper_plt.show()
roc_plt.show()

