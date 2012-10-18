from svmutil import * 
import re

class getInstances():
    def __init__(self):
        self.features = []
        
    def ageLimit(self, age, limit):
        if age < limit:
            self.features.append(1)
        else: self.features.append(0)

    def jobSector(self, job, industry):
        if industry in job:
            self.features.append(1)
        else: self.features.append(0)

    def edLevel(self, degree, relevant_degree):
        if degree == relevant_degree:
            self.features.append(1)
        else:
            self.features.append(0)

    def sexMale(self, sex):
        if sex == 'Male':
            self.features.append(1)
        else: self.features.append(0)

    def nativeCountry(self, country):
        if 'United-States' in country:
            self.features.append(1)
        else: self.features.append(0)
        
    def isWhite(self, race):
        if 'White' in race:
            self.features.append(1)
        else: self.features.append(0)

    def instances(self):
        return self.features

class Learn():
    def __init__(self, infile):
        self.infile = infile
        self.instance_list = []
        self.label_list = []

    def getLabelList(self,label):
        if ">50K" in label:
            return 1
        else: return 0

    def study(self):
        for line in open(self.infile).readlines():
            data = line.split(',')
            i = getInstances()
            i.ageLimit(float(data[0]),40)
            i.ageLimit(float(data[0]),50)
            i.ageLimit(float(data[0]),60)
            i.jobSector(data[1],'Private')
            i.jobSector(data[1],'State-gov')
            i.edLevel(data[3],'Bachelors')
            i.edLevel(data[3],'Masters')
            i.edLevel(data[3],'Doctorate')
            i.sexMale(data[9])
            i.nativeCountry(data[-2])
            i.isWhite(data[8])
            self.instance_list.append(i.instances())

            label = self.getLabelList(data[-1])
            self.label_list.append(label)
        
        return self.label_list, self.instance_list

    def getModel(self):
        prob = svm_problem(self.label_list, self.instance_list)
        m = svm_train(prob)
        return m

train= Learn('adult.data')    
test = Learn('adult.test')
labels, instances = train.study()
model = train.getModel()
test_labels,test_instances = test.study()
output_labels,acc,vals = svm_predict(test_labels,test_instances,model)
