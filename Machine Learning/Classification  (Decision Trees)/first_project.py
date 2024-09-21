
import pandas as pd
import numpy as np

heart_dataset = "heart.csv"
glass_dataset = "glass.csv"
# adjust these 3 parameters to avoid overfitting
heart_treshold = .7 # must be less than .9
glass_treshold = .3 #must be less than .9
depth_treshold = 10
def entropy(targets):
    options = {}
    for t in targets:
        if not t in options.keys():
            options[t] = 1
        options[t] +=1
    ent = 0
    for key in options.keys():
        ent -= (options[key] / len(targets)) * np.log2(options[key] / len(targets))
    return ent
class Node:
    def __init__(self , attr , father_attribute_value):
        self.father_attribute_value = father_attribute_value
        self.attribute = attr
        self.children = []
        self.is_leaf = False

        
class DecisionTree:
    def __init__(self,dataset,target_column , class_count,treshold):
        self.root = Node("" , None)
        # this is reference and will not change
        self.dataset = dataset
        self.target_column = target_column
        self.class_count = class_count
        self.test_data = []
        self.treshold = treshold
        pass
    def get_attribute_values(self,attribute,dataset):
        values = {}
        for set in dataset[attribute]:
            if not set in values.keys():
                values[set] = 0
            values[set] += 1
        return values

    def set_root(self,attr):
        self.root.attribute = attr
        self.root.father_attribute_value = None
    def set_test_data(self , test_data):
        self.test_data = test_data
    def draw(self,indent = 0,node = None):
        if node is None:
            node = self.root
        indent_str = ""
        for i in range(indent):
            indent_str += "\t"
        print(indent_str +str(node.father_attribute_value)+"--->"+ str(node.attribute))
        for n in node.children:
            self.draw(indent + 1,n)
        pass
    def estimate(self , test_data , i , attribute , node):
        if node.is_leaf:
            if node.attribute == test_data[self.target_column][i]:
                return True
            else:
                return False
        for child in node.children:
            if test_data[attribute][i] >= child.father_attribute_value:
                # checking for leaf
                if child.is_leaf:
                    if child.attribute == test_data[self.target_column][i]:
                        return True
                    else:
                        return False
                return self.estimate(test_data,i,child.attribute,child)
    def get_accuracy(self):
        root_attribute = self.root.attribute
        true_count = 0
        for i in range(len(self.test_data[self.target_column])):
            if self.estimate(self.test_data,i,root_attribute,self.root) == True:
                true_count+=1
        return true_count * 100 / len(self.test_data[self.target_column])
    def check_leaf(self,dataset,old_dataset,depth):
        max_target_dict = {}
        max_target_count = -1
        max_target_label = ""
        total_count = len(dataset[self.target_column])
        if total_count == 0:
            dataset = old_dataset
        for target in dataset[self.target_column]:
            if not target in max_target_dict.keys():
                max_target_dict[target] = 0
            max_target_dict[target] += 1
            if max_target_dict[target] > max_target_count:
                max_target_count = max_target_dict[target]
                max_target_label = target
        total_count = len(dataset[self.target_column])
        # print("total count is : "+ str(total_count))
        # division zero protection
        # if total_count == 0:
        #     old_dataset.reset_index(drop=True,inplace=True)
        #     return True , old_dataset[self.target_column][0]
        # limit = self.treshold / self.class_count
        value = (max_target_count / total_count)
        if  value > self.treshold:
            return True , max_target_label
        if depth >= depth_treshold:
            return True , max_target_label
        return False , None



    def build_tree(self,dataset,columns,node,depth=0):
        i = 1
        for child in node.children:
            # filtering the dataset
            old_dataset = dataset.copy()
            dataset.drop(dataset[dataset[node.attribute] == child.father_attribute_value].index,inplace=True)
            if depth == 0:
                print(str(i)+" of "+str(len(node.children))+" root subtrees has been passed")
            is_leaf , label = self.check_leaf(dataset,old_dataset,depth)
            if is_leaf:
                child.attribute = label
                child.is_leaf = True
                i+=1
                continue
            # current_target = dataset[self.target_column]
            # is_leaf = True
            # for target in dataset[self.target_column]:
            #     if(target != current_target):
            #         is_leaf = False
            # if is_leaf:
            #     child.attribute = current_target
            #     child.is_leaf = True
            # filtering the values

            # set child attribute here
            max_ig = -1;
            max_attribute = ""
            for attribute in columns:
                if attribute != self.target_column:
                    ig = self.information_gain(dataset,attribute)
                    if ig > max_ig:
                        max_attribute = attribute
                        max_ig = ig
            child.attribute = max_attribute
            columns_copy = columns.copy()
            columns_copy.remove(child.attribute)
            values = self.get_attribute_values(max_attribute,dataset)
            for value in values.keys():
                child.children.append(Node("",father_attribute_value=value))
            self.build_tree(dataset.copy(),columns_copy,child,depth+1)
            i+=1
        pass

    def learn_tree(self):
        #choosing best attribute for root
        max_ig = -1;
        max_attribute = ""
        for attribute in self.dataset.columns:
            if attribute != self.target_column:
                ig = self.information_gain(self.dataset,attribute)
                if ig > max_ig:
                    max_attribute = attribute
                    max_ig = ig
        self.set_root(max_attribute)
        values = self.get_attribute_values(max_attribute,self.dataset.copy())
        for value in values.keys():
            self.root.children.append(Node("",father_attribute_value=value))
        columns = self.dataset.columns.tolist().copy()
        columns.remove(self.target_column)
        columns.remove(max_attribute)
        self.build_tree(self.dataset.copy(),columns,self.root)
        pass
    def information_gain(self ,dataset, attribute):
        # getting values of attribute
        values = self.get_attribute_values(attribute,dataset)
        #calculating sigma
        sigma = 0
        for key in values.keys():
            sigma += (values[key] / len(dataset[attribute])) * entropy(dataset[self.target_column])
        return entropy(self.dataset[self.target_column]) - sigma

def get_classes(targets , index):
    k1 = 0
    k2 = 0
    seen_targets = []
    for i in range(index):
        if not targets[i] in seen_targets:
            k1+=1
            seen_targets.append(targets[i])
    seen_targets = []
    for i in range(index+1,len(targets)):
        if not targets[i] in seen_targets:
            k2+=1
            seen_targets.append(targets[i])
    return k1 , k2
def check_alpha_point(dataset,column , target_column , K ,alpha_point):
    N = len(dataset[column])
    # finding cut index
    cut_index = 0
    for i in range(len(dataset[column])):
        if dataset[column][i] > alpha_point:
            cut_index = i
            break
    all_targets = dataset[target_column].to_numpy()
    left_side_targets = dataset[target_column].to_numpy()[0:cut_index-1]
    right_side_targets = dataset[target_column].to_numpy()[cut_index:]
    k1 , k2 = get_classes(all_targets,cut_index)
    return 0 < N*entropy(all_targets) + K * entropy(all_targets) - len(left_side_targets) * entropy(left_side_targets) - len(right_side_targets) * entropy(right_side_targets) - np.log2(N-1) - np.log2((3**K)-2) - k1 * entropy(left_side_targets) - k2 * entropy(right_side_targets)

def find_alpha_cut_points(dataset,column,target_column):
    alpha_points = []
    values = [dataset[column] , dataset[target_column]]
    current_target = values[1][0]
    for i in range(len(values[0])):
        if values[1][i] != current_target:
            # toggle current target
            current_target = values[1][i]
            # add possible alpha point
            alpha_points.append((values[0][i-1] + values[0][i]) /2)
    return alpha_points

    
def discritize(dataset,column,target_column,K):
    dataset.sort_values(by=[column],inplace=True)
    dataset.reset_index(drop=True,inplace=True)
    #finding alpha cut points
    alpha_points = find_alpha_cut_points(dataset,column,target_column)
    alpha_points_copy = alpha_points.copy()
    for alpha_point in alpha_points_copy:
        if check_alpha_point(dataset,column,target_column,K,alpha_point) == False:
            alpha_points.remove(alpha_point)
    alpha_points.insert(0,dataset[column][0])
    temporary_alpha_point_index=1
    for i in range(len(dataset[column])):
        if temporary_alpha_point_index < len(alpha_points):
            if(dataset[column][i] >= alpha_points[temporary_alpha_point_index]):
                temporary_alpha_point_index+=1
            dataset[column][i] = alpha_points[temporary_alpha_point_index-1]
        else:
            dataset[column][i] = alpha_points[len(alpha_points)-1]
    return dataset
             
heart_data = pd.read_csv(heart_dataset)
glass_data = pd.read_csv(glass_dataset)
# four columns need to be discretized : age,trestbps,chol,thalach
heart_columns = ['age','trestbps','chol','thalach']
for column in heart_columns:
    heart_data = discritize(heart_data,column,'target',2)
glass_columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
for column in glass_columns:
    glass_data = discritize(glass_data , column , 'Type' , 6)
heart_data.sort_values(by=heart_columns , inplace=True)
glass_data.sort_values(by=glass_columns ,inplace=True)
# printing discretized data
print(heart_data)
print(glass_data)
# split dataset
heart_train_data=heart_data.sample(frac=0.7,random_state=200)
heart_test_data=heart_data.drop(heart_train_data.index)
glass_train_data=glass_data.sample(frac=0.7 , random_state=200)
glass_test_data=glass_data.drop(glass_train_data.index)
# reseting indices
heart_test_data.reset_index(drop=True , inplace=True)
glass_test_data.reset_index(drop=True , inplace=True)
# running ID3 Algorithm
heart_tree = DecisionTree(heart_train_data,"target",2,heart_treshold)
heart_tree.learn_tree()
heart_tree.set_test_data(heart_test_data)
glass_tree = DecisionTree(glass_train_data,"Type",6,glass_treshold)
glass_tree.learn_tree()
glass_tree.set_test_data(glass_test_data)
print("heart dataset DT: ")
heart_tree.draw()
print("glass dataset DT: ")
glass_tree.draw()
print("heart dataset accuracy : ")
print(heart_tree.get_accuracy())
print("glass dataset accuracy : ")
print(glass_tree.get_accuracy())

