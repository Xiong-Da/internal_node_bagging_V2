import tensorflow as tf

from DropConvs import *

######################################################################################
#model for 32*32 RGB images

BASE_WEIDTH=[64,64,128,128,256,256]
def getNumOfFilters(layerIndex,modelWidth):
    return int(BASE_WEIDTH[layerIndex]*modelWidth)

FILTER_SIZE=[3,3,3,3,3,1]
def getFilterSize(layerIndex):
    return FILTER_SIZE[layerIndex]

VALID_PADDING_LAYER=[4]
def getPaddingType(layerIndex):
    if layerIndex in VALID_PADDING_LAYER:
        return "VALID"
    return "SAME"

POOL_LAYER=[1,3]
def poolingFeature(output, layerIndex):
    if layerIndex in POOL_LAYER:
        output = tf.layers.max_pooling2d(output, [3, 3], [2, 2],padding="SAME", name="max_pool" + str(layerIndex))

    return output

def getSmallRGBImageModel(image,modelWidth,classNum,activateFun,dropoutType, groupNum, keepProb,isTraining):
    modelDepth=len(BASE_WEIDTH) #fix model size

    combineOps = []
    averageOps = []

    output = image

    if dropoutType == "probOut":
        convFun = probOutConv
    elif dropoutType == "singleOut":
        convFun = singleOutConv
    elif dropoutType == "continueProbOut":
        convFun = continueProbOutConv
    else:
        raise Exception("unkown droptype:" + dropoutType)

    for layerIndex in range(modelDepth):
        filterNum = getNumOfFilters(layerIndex,modelWidth)
        filterSize=getFilterSize(layerIndex)
        paddingType=getPaddingType(layerIndex)


        output, _combineOps, _averageOps = convFun(
            output, filterSize, filterNum, 1, paddingType, True,activateFun,
            groupNum, keepProb,
            isTraining,
            dropoutType+"Conv_" + str(layerIndex)
        )

        combineOps += _combineOps
        averageOps += _averageOps

        output = poolingFeature(output, layerIndex)

    convShape = output.shape.as_list()
    #print("last feature map size:"+str(convShape))
    output = tf.layers.average_pooling2d(output, convShape[1], convShape[1])

    output = tf.reshape(output, [-1, convShape[-1]])

    return tf.layers.dense(output, classNum), combineOps, averageOps

##############################################################################

def getFullConnectedModel(image,modelWidth,classNum,activateFun,dropoutType, groupNum, keepProb,isTraining):
    imageShape=image.shape.as_list()
    image = tf.reshape(image, [-1, 1, 1, imageShape[1] * imageShape[2]*imageShape[3]])

    modelDepth=2
    modelWidth = int(modelWidth)

    combineOps = []
    averageOps = []

    output = image

    if dropoutType in [ "probOut", "singleOut","continueProbOut"]:
        if dropoutType == "probOut":
            convFun = probOutConv
        if dropoutType == "singleOut":
            convFun = singleOutConv
        if dropoutType == "continueProbOut":
            convFun = continueProbOutConv

        for i in range(modelDepth):
            output, _combineOps, _averageOps = convFun(
                output, 1, modelWidth, 1, "SAME", True, activateFun,
                groupNum, keepProb,
                isTraining,
                dropoutType + "Conv_" + str(1)
            )
            combineOps += _combineOps
            averageOps += _averageOps
    else:
        raise Exception("unsupported dorpout type:" + dropoutType)

    output = tf.reshape(output, [-1, modelWidth])

    return tf.layers.dense(output, classNum), combineOps, averageOps