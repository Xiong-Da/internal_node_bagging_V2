import tensorflow as tf
import math

#####################################################################################
def getFilter(input,filterSize,filterNum):
    tensorShape = input.get_shape().as_list()
    inputChannel = tensorShape[-1]  # assume channel last

    # MSRA initialization
    filter = tf.Variable(tf.truncated_normal([filterSize, filterSize, inputChannel, filterNum], 0.0,
                                             stddev=math.sqrt(2.0 / (inputChannel * filterSize * filterSize))))
    return filter

#############################################################################################

class GroupedProbout():
    def __init__(self,prob):
        self.keepProb=prob

    def drop(self,outputs):
        with tf.variable_scope("probOut"):
            droped=[]
            for output in outputs:
                droped.append(tf.nn.dropout(output,self.keepProb))
        return tf.add_n(droped)

    def average(self,averageVariable,variables):
        combineOps = []
        averageOps = []

        combineOp = tf.assign(averageVariable, tf.add_n(variables)) #bc use tf.nn.dropout above,so dont multi with keepprob
        combineOps.append(combineOp)

        with tf.control_dependencies(combineOps):
            for variable in variables:
                op = tf.assign(variable, averageVariable/len(variables))
                averageOps.append(op)

        return combineOps,averageOps

from tensorflow.python.ops import array_ops,random_ops
class GroupedSingleout():
    def __init__(self):
        pass

    def drop(self, outputs):
        with tf.variable_scope("singleOut"):
            #randomOfMusk = tf.random_uniform(tensorShape, minval=0, maxval=1)
            randomOfMusk = random_ops.random_uniform(array_ops.shape(outputs[0]),minval=0, maxval=1 ) #decouple with batch size

            musks = []
            for i in range(len(outputs)):
                lowProbBound = i * 1.0 / len(outputs)
                upProbBoud = (i + 1) * 1.0 / len(outputs)

                isKeep = tf.logical_and(randomOfMusk >= lowProbBound, randomOfMusk < upProbBoud)
                isKeep = tf.cast(isKeep, tf.float32)

                musks.append(isKeep)

            muskedOutputs = []
            for i in range(len(musks)):
                muskedOutputs.append(outputs[i] * musks[i])
            output=tf.add_n(muskedOutputs)

        return output

    def average(self, averageVariable, variables):
        combineOps = []
        averageOps = []

        combineOp = tf.assign(averageVariable, tf.add_n(variables)/ len(variables))
        combineOps.append(combineOp)

        with tf.control_dependencies(combineOps):
            for variable in variables:
                op = tf.assign(variable, averageVariable)
                averageOps.append(op)

        return combineOps, averageOps

class GroupedContinueProbout():
    def __init__(self,prob):
        self.keepProb=prob

    def drop(self, outputs):
        with tf.variable_scope("ContinueProbout"):
            musks = []
            for i in range(len(outputs)):
                #there are 3 diff type of noise musk, all E[musk]=keepProb
                continueMusk = random_ops.random_uniform(array_ops.shape(outputs[0]), minval=0, maxval=self.keepProb*2)
                # continueMusk = random_ops.random_normal(array_ops.shape(outputs[0]), self.keepProb,math.sqrt(self.keepProb * (1 - self.keepProb)))
                # continueMusk = tf.minimum(continueMusk, self.keepProb * 2) #keep E(mask)=gropNum*keepProb
                # continueMusk = tf.maximum(continueMusk, 0.0)

                musks.append(continueMusk)

            muskedOutputs = []
            for i in range(len(musks)):
                muskedOutputs.append(outputs[i] * musks[i])
            output=tf.add_n(muskedOutputs)

        return output

    def average(self, averageVariable, variables):
        combineOps = []
        averageOps = []

        combineOp = tf.assign(averageVariable, tf.add_n(variables)*self.keepProb)
        combineOps.append(combineOp)

        with tf.control_dependencies(combineOps):
            for variable in variables:
                op = tf.assign(variable, averageVariable/(len(variables)*self.keepProb))
                averageOps.append(op)

        return combineOps, averageOps

# class GroupedScaleout():
#     def __init__(self,keepProb,groupNum):
#         self.keepProb=keepProb
#         self.groupNum=groupNum
#
#     def drop(self, outputs):
#         if len(outputs) != 1:
#             raise Exception("scale out just need one input to drop function")
#
#         with tf.variable_scope("scaleOut"):
#             scaleMusk=random_ops.random_normal(array_ops.shape(outputs[0]),self.keepProb * self.groupNum,
#                                          math.sqrt(self.keepProb * (1 - self.keepProb) * self.groupNum))
#             scaleMusk = tf.minimum(scaleMusk, self.groupNum * self.keepProb * 2) #keep E(mask)=gropNum*keepProb
#             scaleMusk = tf.maximum(scaleMusk, 0.0)
#
#         return outputs[0] * scaleMusk
#
#     def average(self, averageVariable, variables):
#         combineOps = []
#         averageOps = []
#
#         combineOp = tf.assign(averageVariable, tf.add_n(variables)*(self.groupNum*self.keepProb))
#         combineOps.append(combineOp)
#
#         return combineOps,averageOps
#
# class GroupedNormalout():
#     def __init__(self,keepProb):
#         self.keepProb=keepProb
#
#     def drop(self, outputs):
#         if len(outputs) != 1:
#             raise Exception("normal out just need one input to drop function")
#
#         return tf.nn.dropout(outputs[0],self.keepProb)
#
#     def average(self, averageVariable, variables):
#         combineOps = []
#         averageOps = []
#
#         combineOp = tf.assign(averageVariable, tf.add_n(variables))
#         combineOps.append(combineOp)
#
#         return combineOps,averageOps


def groupedDropoutConv(input,filterSize,filterNum,stride,padding,isUsingBias,activateFun,
                          groupNum,isTraing,dropHandler,
                          name):
    with tf.variable_scope(name):
        #define variable
        with tf.variable_scope("average"):
            averageFilter=getFilter(input,filterSize,filterNum)
            if isUsingBias==True:
                averageBias=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[filterNum]))

        filters = []
        biass = []
        with tf.variable_scope("kernel"):
            for _ in range(groupNum):
                filters.append(getFilter(input, filterSize, filterNum))
        if isUsingBias == True:
            with tf.variable_scope("bias"):
                for _ in range(groupNum):
                    biass.append(tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[filterNum])))

        def inferenceFun():
            output = tf.nn.conv2d(input, averageFilter, [1,stride,stride,1], padding)
            if isUsingBias == True:
                output = tf.nn.bias_add(output, averageBias)
            return activateFun(output)

        def trainFun():
            features = []
            for i in range(groupNum):
                output = tf.nn.conv2d(input, filters[i], [1, stride, stride, 1], padding)
                if isUsingBias == True:
                    output = tf.nn.bias_add(output, biass[i])
                output = activateFun(output)
                features.append(output)

            return dropHandler.drop(features)

        output=tf.cond(isTraing,trainFun,inferenceFun)

        #ops below should be run manually in training time
        combineOps=[]   #ops to store combined weights n->1
        averageOps=[]    #ops to assign average weight to corresponding weights 1->n

        _combineOps,_averageOps=dropHandler.average(averageFilter,filters)
        combineOps+=_combineOps
        averageOps+=_averageOps

        if isUsingBias==True:
            _combineOps, _averageOps = dropHandler.average(averageBias, biass)
            combineOps += _combineOps
            averageOps += _averageOps

        return output,combineOps,averageOps

def probOutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
                groupNum,keepProb,
                isTraing,
                name):
    probOutHandler=GroupedProbout(keepProb)
    return groupedDropoutConv(input,filterSize,filterNum,stride,padding,isUsingBias,activateFun,
                          groupNum,isTraing,probOutHandler,
                          name)

def continueProbOutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
                groupNum,keepProb,
                isTraing,
                name):
    continueProboutHandler=GroupedContinueProbout(keepProb)
    return groupedDropoutConv(input,filterSize,filterNum,stride,padding,isUsingBias,activateFun,
                          groupNum,isTraing,continueProboutHandler,
                          name)

def singleOutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
                  groupNum,keepProb,
                  isTraing,
                  name):
    singleOutHandler = GroupedSingleout()
    return groupedDropoutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
                              groupNum, isTraing, singleOutHandler,
                              name)

# def scaleOutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
#                  groupNum,keepProb,
#                  isTraing,
#                  name):
#     #groupNum and keepProb decide how to scale node output
#     scaleOutHandler = GroupedScaleout(keepProb,groupNum)
#
#     #set group size to one, random variable is handled in scaleOutHandler
#     return groupedDropoutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
#                               1, isTraing, scaleOutHandler,
#                               name)
#
# def normalOutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
#                  groupNum,keepProb,
#                  isTraing,
#                  name):
#     #groupNum and keepProb decide how to scale node output
#     normalOutHandler = GroupedNormalout(keepProb)
#
#     #set group size to one, random variable is handled in normalOutHandler
#     return groupedDropoutConv(input, filterSize, filterNum, stride, padding, isUsingBias,activateFun,
#                               1, isTraing, normalOutHandler,
#                               name)