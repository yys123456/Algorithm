from k_means import *
from apriori import *
from newFP import *
from SVM import *
from k_medoids import *
import pandas as pd
import xlrd
import numpy as np
from DFTools import *
from getRules import *
class process:
	def __init__(self,fileName,method,paras):
		self.method=method
		self.paras=paras
		self.DF=getDF(fileName)
		self.preTreat()

	def start(self):
		paras=self.paras
		if self.method==1:
			return self.apriori(self.dataList,paras[0],paras[1])#paras依次：min_sup,min_conf
		if self.method==2:
			return self.FP_Growth(self.dataList,paras[0],paras[1])#paras依次：min_sup,min_conf
		if self.method==3:
			return self.k_means(self.dataList,paras[0],paras[1])#paras依次：k,最大迭代次数
		if self.method==4:#SVM
			return self.SVM(self.dataList,paras)#paras依次：惩罚系数、核函数、需要绘制ROC
		if self.method==5:#k_medoids
			return self.k_medoids(self.dataList,paras[0])#只有paras[0]：k


		
	def apriori(self,dataList,min_sup,min_conf):
		ap=apriori(dataList,min_sup,min_conf)
		ap.start()
		freq=ap.getOutCome()#频繁项集
		rules=rule_gen(freq,len(dataList),min_conf)
		return freq,rules

	def k_means(self,dataList,k,iteration):
		km=k_means(k,iteration,dataList)
		cif=tranDF2list(km.getOutPut())
		plt=km.getImg()
		return cif,plt


	def FP_Growth(self,dataList,min_sup,min_conf):
		freq,tree=FP_Growth(dataList,min_sup)
		rules=rule_gen(freq,len(dataList),min_conf)
		treeplt=getTreePlt(tree)
		return freq,rules,treeplt


	def SVM(self,dataList,paras):#使用SVM进行二分类
		penalty_coefficient=paras[0]#惩罚系数
		kernal=paras[1]#核函数
		need_ROC=paras[2]#是否需要绘制ROC
		svm=None
		if need_ROC==1:
			svm=SVM(penalty_coefficient,kernal,dataList[0],dataList[1],dataList[2])
		else:
			svm=SVM(penalty_coefficient,kernal,dataList[0],dataList[1])
		svm.train()
		plt_raw=svm.getPtsImg()
		plt_Hyper=svm.getHyperImg()
		result=svm.getOutCome()		
		if need_ROC==1:
			ROC=svm.getROC()
			return result,plt,plt_Hyper,ROC
		else:
			return result,plt,plt_Hyper

	def k_medoids(self,dataList,k):
		kct=k_medoids(dataList,k)
		kct.start()
		cif=kct.getOutPut()
		plt=kct.getImg()
		return cif,plt


	def preTreat(self):
		if self.method in [1,2]:#apriori fp
			self.dataList=tranDF2list(self.DF[0])#只用第一个sheet
		elif self.method == 4:#SVM分类
			if self.paras[2]==1:#需要ROC曲线
				self.dataList=[tranDF2list(self.DF[0]),tranDF2list(self.DF[1]),tranDF2list(self.DF[2])]
			else:#不需要ROC曲线
				self.dataList=[tranDF2list(self.DF[0]),tranDF2list(self.DF[1])]
		elif self.method==5:
			self.dataList=tranDF2list(self.DF[0])#Kmedoid
		else:
			self.dataList=self.DF[0]#kmeans
	





# print(DF)
# dataList=[]
# for i in range(1,sheet.nrows):
#     arr=[]
#     for j in range(1,sheet.ncols):
#         if sheet.cell(i,j).value!='':
#            arr.append(sheet.cell(i,j).value)
#     dataList.append(arr)

# dataList为一个二维list
# if flag==1:#此时使用apriori进行关联性分析
# 	dataList=tranDF2list(DF)
# 			print(dataList)
# 			ap=apriori(dataList,2/6,0.6)
# 			ap.start()
# 			result=ap.getOutCome();
# 			ap.rule_gen()
# 			l,r=ap.getRule()
# 			for i in result:
# 				print(i)
# 			for key in l:
# 				print(l[key],"=>",r[key])
# if flag==2:#使用fpgrowth方法进行关联性分析
# 	dataList=tranDF2list(DF)
# 	fp=FP_Growth(dataList,2,0.6)
# 	# fp.createTree()
# 	freq,treeDic=fp.get_freq_sets()
# 	print(freq)
# 	print(treeDic)
# 	# print(treeDic)
# 	# print(fp.get_FP_Tree())
# if flag==3:#使用k-means聚类 最多支持二维数据聚类返回图片，高维不返回图片
# 	km=k_means(3,3,DF)
# 	print(km.getOutPut())
# 	km.getImg().show()

#SVM和PAM


