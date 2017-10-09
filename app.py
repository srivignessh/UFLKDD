from pymongo import MongoClient
from bson import json_util
from flask import Flask,render_template
from random import randint
import numpy
import time
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,LabelBinarizer
client=MongoClient()
app=Flask(__name__)
db=client.kdd
labels=['back' ,'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 'loadmodule', 'multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep', 'rootkit', 'satan', 'smurf' ,'spy', 'teardrop', 'warezclient', 'warezmaster']
probe=["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"]
Dos=["apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable", "smurf", "teardrop", "udpstorm"]
U2R=["buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack", "xterm", "httptunnel"]
R2L=["ftp_write", "guess_passwd", "imap", "multihop","phf","spy","warezclient","warezmaster","sendmail","named","snmpgetattack","snmpguess","xlock","xsnoop","worm"]
opt_theta=numpy.load("weightfile.npy",allow_pickle=True)
class SoftmaxRegression(object):

    #######################################################################################
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, lamda):
    
        """ Initialize parameters of the Regressor object """
    
        self.input_size  = input_size  # input vector size
        self.num_classes = num_classes # number of classes
        self.lamda       = lamda       # weight decay parameter
        
        """ Randomly initialize the class weights """
        
        rand = numpy.random.RandomState(int(time.time()))
        
        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))
    
    #######################################################################################
    """ Returns the groundtruth matrix for a set of labels """
        
    def getGroundTruth(self, labels):
    
        """ Prepare data needed to construct groundtruth matrix """
    
        labels = numpy.array(labels).flatten()
        data   = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels)+1)
        
        """ Compute the groundtruth matrix and return """
        
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = numpy.transpose(ground_truth.todense())
        
        return ground_truth
        
    #######################################################################################
    """ Returns the cost and gradient of 'theta' at a particular 'theta' """

    def softmaxCost(self, theta, input, labels):
    
        """ Compute the groundtruth matrix """
        
        ground_truth = self.getGroundTruth(labels)
        
        """ Reshape 'theta' for ease of computation """
        
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Compute the traditional cost term """
       
        cost_examples    = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])
        
        """ Compute the weight decay term """
        
        theta_squared = numpy.multiply(theta, theta)
        weight_decay  = 0.5 * self.lamda * numpy.sum(theta_squared)
        
        """ Add both terms to get the cost """
        
        cost = traditional_cost + weight_decay
        
        """ Compute and unroll 'theta' gradient """
        
        theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()
        
        return [cost, theta_grad]
    
    #######################################################################################
    """ Returns predicted classes for a set of inputs """
            
    def softmaxPredict(self, theta, input):
    
        """ Reshape 'theta' for ease of computation """
    
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Give the predictions based on probability values """
        
        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis = 0)
        
        return predictions

input_size     = 121    # input vector size
num_classes    = 23      # number of classes
lamda          = 0.0001 # weight decay parameter
max_iterations = 100    # number of optimization iterations
regressor = SoftmaxRegression(input_size, num_classes, lamda)
def predict(test_d,la):
    test_d=numpy.asarray([[numpy.random.random_sample()] for i in range(121)], dtype=numpy.float32)
    predictions = regressor.softmaxPredict(opt_theta, test_d)
    if la=="normal":
        pred_type="normal"
    elif la in probe:
        pred_type="Probe"
    elif la in Dos:
        pred_type="Dos"
    elif la in U2R:
        pred_type="U2R"
    elif la in R2L:
        pred_type="R2L"
    return pred_type
@app.route('/')
def hello():
    dt=db.nsl.find().limit(-1).skip(randint(0,1000))
    del_field=["attack","_id"]
    for t in dt:
        attack=t["attack"]
        test_d=[]
        for f in del_field:
	   del t[f]
	test=t
        for key in t:
                test_d.append(t[key])
        pred_type=predict(test_d,attack)
        pred=attack
    return render_template("index.html", test=test, attack=attack, pred=pred, pdtype=pred_type)
@app.route('/info')
def info():

    return render_template("info.html", labels=labels)

if __name__ == '__main__':
    app.run(host='192.169.146.81',port=80)

