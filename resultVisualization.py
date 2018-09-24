
testErrDataMap={}

def loadTestErr(filePath):
    global testErrDataMap
    with open(filePath) as f:
        result=f.read()
        for line in result.splitlines():
            #modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch,minTestErr
            dataMap = testErrDataMap
            datas=line.split(sep=',')[:10]
            for data in datas[:-2]:
                if data not in dataMap.keys():
                    dataMap[data]={}
                dataMap=dataMap[data]
            if datas[-2] not in dataMap.keys():
                dataMap[datas[-2]]=[]
            dataMap[datas[-2]].append(float(datas[-1]))

def _printLowestErr(dataMap,dataStr):
    if type(dataMap) == list:
        print(dataStr+str(min(dataMap)))
    else:
        for key in list(dataMap.keys()):
            _printLowestErr(dataMap[key],dataStr+str(key)+",")

def printLowestErr():
    global testErrDataMap
    _printLowestErr(testErrDataMap,"")

def getTestErr(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch):
    global testErrDataMap
    container=testErrDataMap

    if groupNum==1:
        perAverageEpoch=None

    if dropoutType == "singleOut":
        keepProb=None

    for param in [modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch]:
        container=container[str(param)]
    return min(container)


#########################################################################################################################################

import matplotlib
import matplotlib.pyplot as plt

colorMap = {1: "blue", 2: "green", 4: "orange", 8: "red",
            "sigmoid":"blue","tanh":"green","relu":"orange"}
methodName={"probOut":"method A","singleOut":"method B"}
lineStyleMap={"probOut":'-', "singleOut":'--',64:"-",256:"--"}

FONT_SIZE=14
LINE_WIDTH=3.0

plt.tight_layout()
plt.subplots_adjust(left=0.13,bottom=0.11,right=0.98,top=0.93)

matplotlib.rcParams['ps.useafm']=True
matplotlib.rcParams['pdf.use14corefonts']=True
matplotlib.rcParams['text.usetex']=False

def getErrsOfDiffMethodsWithDiffWidth(modelFunName, activateFunName, datasetName, modelWidthList, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch):
    errs=[]
    for modelWidth in modelWidthList:
        err=getTestErr(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch)
        errs.append(err)
    return errs

def visualizeDiffMethods():
    #modelWidthList=[64,128,256,512,1024]
    modelWidthList = [64, 128, 256, 512, 1024]
    dropoutTypeList=["probOut", "singleOut"]
    groupNumList=[1,2,4]
    keepProb=0.5
    batchSize=256
    perAverageEpoch=10
    
    dropType="singleOut"
    for groupNum in groupNumList:
        curveName="group size "+str(groupNum)
        y=getErrsOfDiffMethodsWithDiffWidth("fullConnected","relu","MNIST",
                                            modelWidthList,dropType,groupNum,keepProb,batchSize,perAverageEpoch)
        print(curveName+":"+str(y))
        plt.plot(modelWidthList,y,color=colorMap[groupNum],label=curveName,linewidth=LINE_WIDTH)

    plt.ylim(0.01,0.03)
    plt.xticks(modelWidthList, fontsize=FONT_SIZE)
    plt.yticks([0.01, 0.02, 0.03], fontsize=FONT_SIZE)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("model width", fontsize=FONT_SIZE)
    plt.title("MNIST", fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()

def getErrsOfDiffMethodsWithDiffAverage(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpochList):
    errs=[]
    for perAverageEpoch in perAverageEpochList:
        err=getTestErr(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch)
        errs.append(err)
    return errs

def visualizeWeightAverage():
    modelWidthList = [64, 256]
    dropoutTypeList = ["probOut", "singleOut"]
    groupNumList = [ 2, 4]
    keepProb = 0.5
    batchSize = 256
    perAverageEpochList = [5,10,20,50,100,200]

    modelWidth=256

    for dropType in dropoutTypeList:
        for groupNum in groupNumList:
            curveName = methodName[dropType]+", group size " + str(groupNum)
            y = getErrsOfDiffMethodsWithDiffAverage("fullConnected", "relu", "MNIST",
                                                  modelWidth, dropType, groupNum, keepProb, batchSize, perAverageEpochList)
            print(curveName + ":" + str(y))
            plt.plot(perAverageEpochList, y, color=colorMap[groupNum], label=curveName, linewidth=LINE_WIDTH,linestyle=lineStyleMap[dropType])

    plt.ylim(0.01, 0.03)

    plt.xticks(perAverageEpochList[1:], fontsize=FONT_SIZE-2)
    plt.yticks([0.01,0.02,0.03], fontsize=FONT_SIZE-2)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("weight average frequency(epochs)", fontsize=FONT_SIZE)
    plt.title("MNIST dataset, model width "+str(modelWidth), fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()


from test import getDataSaveDir
import os

def getConvergeData( modelWidth, dropType, groupNum, keepProb, batchSize, perAverageEpoch):
    if groupNum==1:
        perAverageEpoch=None

    if dropType == "singleOut":
        keepProb=None

    logDir = getDataSaveDir("MNIST", modelWidth, "relu", dropType, groupNum, keepProb, perAverageEpoch, batchSize)
    logFileName="MNIST_"+str(modelWidth)+"_"+str(dropType)+"_"+str(groupNum)+"_"+str(keepProb)+"_"+str(perAverageEpoch)+"_"+str(batchSize)+".txt"
    logFilePath=os.path.join("./result",logFileName)
    if os.path.exists(logFilePath)==False:
        oldFilePath=None
        for fileName in os.listdir(logDir):
            if "txt" in fileName:
                oldFilePath=os.path.join(logDir,fileName)
        if oldFilePath is None:
            raise Exception("no data in "+logDir)
        with open(logFilePath,"w") as outF:
            with open(oldFilePath,"r") as inF:
                outF.writelines(inF.readlines())

    x=[]
    y=[]

    with open(logFilePath,"r") as f:
        for line in f.readlines():
            data=line.split()
            x.append(int(data[0]))
            y.append(float(data[1]))

    return x,y


def visualizeCoverge():
    modelWidthList = [64, 256]
    dropoutTypeList = ["probOut", "singleOut"]
    groupNumList = [1, 2, 4]
    keepProb = 0.5
    batchSize = 256
    perAverageEpoch = 20

    modelWidth = 256

    for dropType in dropoutTypeList:
        for groupNum in groupNumList:
            curveName = methodName[dropType]+", group size " + str(groupNum)
            x,y = getConvergeData( modelWidth, dropType, groupNum, keepProb, batchSize, perAverageEpoch)
            plt.plot(x, y, color=colorMap[groupNum], label=curveName, linewidth=LINE_WIDTH,linestyle=lineStyleMap[dropType])

    plt.ylim(0.01, 0.03)

    plt.yticks([0.01,0.02,0.03], fontsize=FONT_SIZE-2)
    plt.xticks([0,10000,20000,30000,40000], fontsize=FONT_SIZE-2)

    plt.ylabel("test error",fontsize=FONT_SIZE)
    plt.xlabel("train step",fontsize=FONT_SIZE)
    plt.title("MNIST dataset, model width "+str(modelWidth),fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()


def visualizeRGBImgResult():
    dropoutTypeList = ["probOut", "singleOut"]
    datasetList=["CIFAR_10","SVHN"]

    modelWidthList = [0.25,0.5,1]
    groupNumList = [1, 2, 4]
    keepProb = 0.5
    batchSize = 256

    dropType = "singleOut"
    dataset="SVHN"

    if dataset=="SVHN":
        perAverageEpoch = 10
    else:
        perAverageEpoch = 10

    for groupNum in groupNumList:
        curveName = "group size " + str(groupNum)
        y = getErrsOfDiffMethodsWithDiffWidth("CNN", "relu", dataset,
                                              modelWidthList, dropType, groupNum, keepProb, batchSize, perAverageEpoch)
        print(curveName + ":" + str(y))
        plt.plot(modelWidthList, y, color=colorMap[groupNum], label=curveName, linewidth=LINE_WIDTH)

    if dataset == "SVHN":
        plt.ylim(0.03, 0.06)
        plt.xticks(modelWidthList, fontsize=FONT_SIZE)
        plt.yticks([0.03, 0.04, 0.05, 0.06], fontsize=FONT_SIZE)
    else:
        plt.ylim(0.1, 0.3)
        plt.xticks(modelWidthList, fontsize=FONT_SIZE)
        plt.yticks([0.1,0.15,0.2,0.25,0.3], fontsize=FONT_SIZE)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("model width", fontsize=FONT_SIZE)
    plt.title(dataset, fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()

def getErrOfDiffGroups(modelFunName, activateFunName, datasetName, modelWidth, dropoutType,
                                            groupNumList, keepProb, batchSize, perAverageEpoch):
        errs = []
        for groupNum in groupNumList:
            err = getTestErr(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb,
                             batchSize, perAverageEpoch)
            errs.append(err)
        return errs

def visualizeActivationFun():
    dropoutTypeList = ["probOut", "singleOut"]
    activationFunList=["relu","sigmoid","tanh"]
    groupNumList = [1, 2, 4]
    keepProb = 0.5
    batchSize = 256
    perAverageEpochList = 10
    modelWidth = 256

    for dropType in dropoutTypeList:
        for activationFun in activationFunList:
            curveName = methodName[dropType] + ", " + str(activationFun)
            y = getErrOfDiffGroups("fullConnected", activationFun, "MNIST",
                                                    modelWidth, dropType, groupNumList, keepProb, batchSize,
                                                    perAverageEpochList)
            print(curveName + ":" + str(y))
            plt.plot(groupNumList, y, color=colorMap[activationFun], label=curveName, linewidth=LINE_WIDTH,
                     linestyle=lineStyleMap[dropType])

    plt.ylim(0.012, 0.022)

    plt.xticks(groupNumList, fontsize=FONT_SIZE - 2)
    plt.yticks([0.012, 0.014, 0.016, 0.018, 0.020,0.022], fontsize=FONT_SIZE - 2)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("group size", fontsize=FONT_SIZE)
    plt.title("MNIST dataset, model width " + str(modelWidth), fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE,loc=1)
    plt.show()

##########################################################################################################################################

if __name__ == "__main__":
    #loadTestErr("./result/fix_full2.csv")
    #visualizeDiffMethods()

    # loadTestErr("./result/CIFAR_10.csv")
    # loadTestErr("./result/SVHN.csv")
    loadTestErr("./result/RGB2.csv")
    visualizeRGBImgResult()

    #loadTestErr("./result/wight_average_data.csv")
    #visualizeWeightAverage()

    #loadTestErr("./result/activation_functions.csv")
    #visualizeActivationFun()

    #visualizeCoverge()