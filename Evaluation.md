#  Evaluation

## 分类任务：

###  P、R、Acc、F1等

~~~python
# 所有指标主要是依赖于labels和predicts【这里的两者维度相同！】
# 导入库
from sklearn import metrics 
# train在训练过程中计算准确度
train_acc = metrics.accuracy_score(true, predic) 

# test 计算
labels=["0","1"]
report = metrics.classification_report(labels_all, predict_all, target_names=labels, digits=4)
confusion = metrics.confusion_matrix(labels_all, predict_all)
# 打印出来
print("Precision, Recall and F1-Score...")
print(report)
print("Confusion Matrix...")
print(confusion)
~~~

解释一下函数：

 target_name: 与标签匹配的名称，就是一个字符串列表，在报告中显示；

 digits:用来设置要输出的格式位数，即指定输出格式的精确度；



classification_report结果：

![image-20230321103037249](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230321103037249.png)

* accuracy：准确度（对整体而言的）

* precision和recall：（对每个类别而言的）
* maro avg的中文名称为宏平均，其计算方式为每个类型的P、R的算术平均，我们以F1的macro avg为例，上述输出结果中的macro avg F1值的计算方式为：

macro avg F1 = (0.9974+0.9977)/2=0.9976

* weighted avg的计算方式与micro avg很相似，只不过weighted avg是用每一个类别样本数量在所有类别的样本总数的占比作为权重，因此weighted avg的计算方式为：

weighted avg F1=  (0.9974* 586+0.9977 *647)/1233=0.9976

confusion_matrix结果：

![image-20230321103140407](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230321103140407.png)



###  曲线ROC、PR等

**ROC曲线和AUC**也可以作为评估指标（在样本不平衡的时候可以使用）这是二分类，多分类要另外考虑。

只需要 导入预测值prob 及 其真实值true

~~~python
# 用库：只需要导入预测值及真实结果true
# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 随机生成预测结果、真实值
y_true = np.random.randint(0, 2, size=100) 
y_prob = np.random.rand(100) # 预测概率，0到1之间的小数

# 计算TPR和FPR
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
# 计算AUC
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
~~~

**多分类ROC**

方法一：对每个类别，都将其作为正例，其他类别作为负例（几个类别几条曲线）——似乎可以用于bad case 分析

```python
# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 随机生成预测结果 和 预测概率
y_true = np.random.randint(0, 3, size=100) # 0、1或2
y_prob = np.random.rand(100, 3) # 预测概率，每行表示一个样本在三个类别下的概率

# 方法一：对每个类别，都将其作为正例，其他类别作为负例
plt.figure()
for i in range(3): # 循环三个类别
    y_true_bin = (y_true == i).astype(int) # 将第i个类别作为正例，其他为负例
    y_prob_bin = y_prob[:, i] # 取出第i个类别下的预测概率
    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob_bin) # 算TPR和FPR
    roc_auc = auc(fpr, tpr) # 计算AUC
    plt.plot(fpr, tpr, label='Class %d (area = %0.2f)' % (i, roc_auc)) # 绘制ROC曲线

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class (method 1)')
plt.legend(loc="lower right")
plt.show()
```

方法二：对每个类别，都将其作为正例，其他类别合并为一类负例，并求平均得到总的ROC曲线（合并成一条曲线）

~~~python
# 方法二：对每个类别，都将其作为正例，其他类别合并为一类负例，并求平均得到总的ROC曲线

# 将真实标签和预测概率转换成二进制形式（one-hot编码）
y_true_bin = label_binarize(y_true, classes=[0, 1 ,2]) 
y_prob_bin = y_prob

# 计算每个类别下的TPR和FPR，并求平均得到总的TPR和FPR（macro-average）
fpr_list = []
tpr_list = []
roc_auc_list = []
for i in range(3): # 循环三个类别
    fpr_i ,tpr_i ,_ = roc_curve(y_true_bin[:, i], y_prob_bin[:, i]) # 计算第i个类别下的TPR和FPR 
    fpr_list.append(fpr_i)
    tpr_list.append(tpr_i)
    roc_auc_i = auc(fpr_i ,tpr_i ) # 计算第i个类别下的AUC 
    roc_auc_list.append(roc_auc_i)

fpr_mean = np.unique(np.concatenate(fpr_list)) # 求所有FPR值的并集，并去重排序得到总的FPR值（x轴）
tprs_interp_mean=[] 
roc_auc_mean=0 

for i in range(3): # 循环三个类别 
    tprs_interp_mean.append(np.interp(fpr_mean,fpr_list[i],tpr_list[i])) # 对每个TPR值进行插值得到与总FPR值相对应的TPR值（y轴）
    tprs_interp_mean[-1][0]=0.0 
    roc_auc_mean+=roc_auc_list[i] 

tprs_interp_mean=np.array(tprs_interp_mean).mean(axis=0) # 求所有TPR值的平均得到总的TPR值（macro-average）
tprs_interp_mean[-1]=1.00 
roc_auc_mean/=3 # 求所有AUC值的平均得到总的AUC值（macro-average）

# 绘制总的ROC曲线和每个类别下的ROC曲线
plt.figure()
plt.plot(fpr_mean,tprs_interp_mean,color='navy',label='Mean ROC (area = %0.2f)' % roc_auc_mean) # 绘制总的ROC曲线
for i in range(3): # 循环三个类别
    plt.plot(fpr_list[i],tpr_list[i],label='Class %d (area = %0.2f)' % (i, roc_auc_list[i])) # 绘制每个类别下的ROC曲线

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class (method 2)')
plt.legend(loc="lower right")
plt.show()
~~~



**PR曲线**

只需要 导入预测值prob 及 其真实值true  【未测试，用的时候在改进】

~~~python
# 用库：只需要导入预测值及其 和真实结果true
# 导入需要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# 随机生成预测结果、真实标签
y_true = np.random.randint(0, 2, size=100) 
y_prob = np.random.rand(100) # 预测概率，0到1之间的小数

# thresholds这个也没啥用
p,r,thresholds = precision_recall_curve(y_true, probas_pred)

# 绘制PR曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 0], [1, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc="lower right")
plt.show()
~~~

