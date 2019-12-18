import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # 可视化绘图
from sklearn import svm  # svm支持向量机
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from cmap import *
#前端使用下拉框提供核函数选择，必须是标准的核函数名
class SVM:#需要训练样本和预测样本
	def __init__(self,penalty_coefficient,kernal,trainData,preData,test_target=None):#预测数据可以是list,必须约定最后一列是类别信息
		self.trainData=[x[0:-1] for x in trainData]#训练数据，简单二维list
		self.target=[x[-1] for x in trainData]#训练数据的类别
		self.classes=list(set(self.target))#类别种类
		self.kernal=kernal#核函数选择
		self.penalty_coefficient=penalty_coefficient
		self.svm_cnf=None#分类器
		self.preData=preData#预测样本
		self.test_target=test_target#需要绘制ROC曲线时使用

	def getPtsImg(self):#绘制trainData降维的散点图
		plt.figure(figsize=(6,6))
		df=pd.DataFrame({'classes':self.target})
		tsne=TSNE(n_components=2, learning_rate=100).fit_transform(self.trainData)
		for i in self.classes:
			d=tsne[df['classes']==i]
			# print(d[0],d[1])
			plt.scatter(d[:,0],d[:,1],label=str(i))
		plt.legend(loc = 'best')
		return plt
	def train(self):#使用训练数据和target进行训练
		self.svm_cnf=svm.SVC(kernel=self.kernal, C=self.penalty_coefficient)
		self.svm_cnf.fit(self.trainData,self.target)
		self.result=self.svm_cnf.predict(self.preData)  #使用模型预测值

	def getOutCome(self):
		return self.result

	def getHyperImg(self):
		# if self.kernal=='linear':
		# 	df_target=pd.DataFrame({'classes':self.target})
		# 	df_trainData=pd.DataFrame(self.trainData)
		# 	plt.figure(figsize=(6,6))
		# 	for i in range(len(self.classes)):
		# 		d=df_trainData.values[df_target['classes']==self.classes[i]]
		# 		plt.scatter(d[:,0],d[:,1],label=str(self.classes[i]))
		# 		# if self.target[i]==self.classes[0]:
		# 		# 	plt.scatter(self.trainData[i][0],self.trainData[i][1],label=str(self.classes[0]))
		# 		# else:
		# 		# 	plt.scatter(self.trainData[i][0],self.trainData[i][1],label=str(self.classes[1]))
		# 	for j in self.svm_cnf.support_:
		# 		plt.scatter(self.trainData[j][0],self.trainData[j][1], s=100, c = '', alpha=0.5, linewidth=1.5, edgecolor='red')
		# 	W = self.svm_cnf.coef_#方向向量W
		# 	b =  self.svm_cnf.intercept_#截距项b
		# 	x = np.arange(0,10,0.01)
		# 	y = (W[0][0]*x+b)/(-1*W[0][1])
		# 	plt.scatter(x,y,s=5,marker = 'h')
		# 	plt.legend(loc = 'best')
		# elif self.kernal=='RBF':
		plt=plot_hyperplane(self.svm_cnf,np.int_(self.trainData),np.int_(self.target))
		return plt


	def getROC(self):
		plt.figure(figsize=(6,6))
		fpr,tpr,threshold = roc_curve(self.test_target, self.result) ###计算真正率和假正率
		roc_auc = auc(fpr,tpr) ###计算auc的值
		lw = 2
		plt.plot(fpr, tpr, color='darkorange',
		         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC')
		plt.legend(loc="lower right")
		return plt


def plot_hyperplane(clf, X, y, 
                h=0.02, 
                draw_sv=True):
	# create a mesh to plot in
	plt.figure(figsize=(6,6))
	x_min,x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min,y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
	plt.title("最大超平面示意图")
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

	markers = ['o', 's', '^']
	colors = ['b', 'r', 'c']
	labels = np.unique(y)
	for label in labels:
	    plt.scatter(X[y==label][:, 0], 
	                X[y==label][:, 1], 
	                c=colors[label], 
	                marker=markers[label])
	# 画出支持向量
	if draw_sv:
	    sv = clf.support_vectors_
	    plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
	return plt
