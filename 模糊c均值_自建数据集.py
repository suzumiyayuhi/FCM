import copy
import math
import random
import time
import matplotlib.pyplot as plt
 
global MAX # 用于初始化隶属度矩阵U
MAX = 10000.0
 
global Epsilon  # 结束条件
Epsilon = 1e-6

class ColorSelect:

    # 颜色筛选器
    @staticmethod
    def color_select(index):
        color_dict = {0: 'go', 1: 'ro', 2: 'bo'}
        return color_dict[index]

class PlotUtils:

    # 画出结果散点图
    @staticmethod
    def plot(result):
        # 画出散点图
        plt.title("FCM")
        plt.xlim(-2, 15)
        plt.ylim(-2, 15)
        plt.xlabel("x")
        plt.ylabel("y")

        for i in range(result.__len__()):

            for j in range(result[i].__len__()):
                print(i, end='')
                plt.plot(result[i][j][0], result[i][j][1], ColorSelect.color_select(i))
        plt.show()

    @staticmethod
    def single_plot(centers):

        # 画出类中心点
        plt.title("FCM")
        plt.xlim(-2, 15)
        plt.ylim(-2, 15)
        plt.xlabel("x")
        plt.ylabel("y")
        for i in range(centers.__len__()):
            plt.plot(centers[i][0], centers[i][1],  ColorSelect.color_select(i))
        plt.show()

def print_matrix(list):
	""" 
	以可重复的方式打印矩阵
	"""
	for i in range(0, len(list)):
		print (list[i])
 
def initialize_U(data, cluster_number):
	"""
	这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
	"""
	global MAX
	U = []
	for i in range(0, len(data)):
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):
			dummy = random.randint(1,int(MAX))
			current.append(dummy)
			rand_sum += dummy
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum
		U.append(current)
	return U
 
def distance(point, center):
	"""
	该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。闵可夫斯基距离
	"""
	if len(point) != len(center):
		return -1
	dummy = 0.0
	for i in range(0, len(point)):
		dummy += abs(point[i] - center[i]) ** 2
	return math.sqrt(dummy)
 
def end_conditon(U, U_old):
    """
	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
	"""
    global Epsilon
    for i in range(0, len(U)):
	    for j in range(0, len(U[0])):
		    if abs(U[i][j] - U_old[i][j]) > Epsilon :
			    return False
    return True
 
def normalise_U(U):
	"""
	在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
	"""
	for i in range(0, len(U)):
		maximum = max(U[i])
		for j in range(0, len(U[0])):
			if U[i][j] != maximum:
				U[i][j] = 0
			else:
				U[i][j] = 1
	return U
 
 
def fuzzy(data, cluster_number, m):
	"""
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    输入参数：簇数(cluster_number)、隶属度的因子(m)的最佳取值范围为[1.5，2.5]
	"""
	# 初始化隶属度矩阵U
	U = initialize_U(data, cluster_number)
	# print_matrix(U)
	# 循环更新U
	while (True):
		# 创建它的副本，以检查结束条件
		U_old = copy.deepcopy(U)
		# 计算聚类中心
		C = []
		for j in range(0, cluster_number):
			current_cluster_center = []
			for i in range(0, len(data[0])):
				dummy_sum_num = 0.0
				dummy_sum_dum = 0.0
				for k in range(0, len(data)):
    				# 分子
					dummy_sum_num += (U[k][j] ** m) * data[k][i]
					# 分母
					dummy_sum_dum += (U[k][j] ** m)
				# 第i列的聚类中心
				current_cluster_center.append(dummy_sum_num/dummy_sum_dum)
            # 第j簇的所有聚类中心
			C.append(current_cluster_center)
 
		# 创建一个距离向量, 用于计算U矩阵。
		distance_matrix =[]
		for i in range(0, len(data)):
			current = []
			for j in range(0, cluster_number):
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)
 
		# 更新U
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				dummy = 0.0
				for k in range(0, cluster_number):
    				# 分母
					dummy += (distance_matrix[i][j ] / distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / dummy
 
		if end_conditon(U, U_old):
			print ("已完成聚类")
			break
	U = normalise_U(U)
	return U 

def make_result(U,data):
    result=[]
    result.append([])
    result.append([])
    result.append([])
    temFlag=0
    for i in range(0,len(U)):
        for j in range(0,3):
            if(U[i][j]==1):
                temFlag=j
                break
        result[temFlag].append(data[i])
    PlotUtils.plot(result)        
    return result

def import_data_from_txt(file):
    data = []
    with open(str(file), 'r') as f:
        for line in f:
            current = line.strip().split(",")  #对每一行以逗号为分割，返回一个list
            current_dummy = []
            for j in range(0, len(current)):
                current_dummy.append(float(current[j]))  #current_dummy存放data
            data.append(current_dummy)
    print("加载数据完毕")
    return data
 
if __name__ == '__main__':
	data= import_data_from_txt("dataset.txt")
	start = time.time()

	# 调用模糊C均值函数
	res_U = fuzzy(data , 3 , 2)
	# 制作画图数据
	result = make_result(res_U,data)
	# 计算准确率
	print ("用时：{0}".format(time.time() - start))
