
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

colorMap = {1: "blue", 2: "green", 4: "orange", 8: "red"}
lineStyleMap={"probOut":'-', "singleOut":'--'}

FONT_SIZE=12
LINE_WIDTH=3.0

plt.tight_layout()
plt.subplots_adjust(left=0.12,bottom=0.1,right=0.98,top=0.95)

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
    groupNumList=[1,2,4,8]
    keepProb=0.5
    batchSize=256
    perAverageEpoch=20
    
    dropType="singleOut"
    for groupNum in groupNumList:
        curveName="group size"+str(groupNum)
        y=getErrsOfDiffMethodsWithDiffWidth("fullConnected","relu","MNIST",
                                            modelWidthList,dropType,groupNum,keepProb,batchSize,perAverageEpoch)
        print(curveName+":"+str(y))
        plt.plot(modelWidthList,y,color=colorMap[groupNum],label=curveName,linewidth=LINE_WIDTH)

    plt.ylim(0.01,0.025)
    plt.xticks(modelWidthList)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("model width", fontsize=FONT_SIZE)
    plt.title(dropType, fontsize=FONT_SIZE)

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
            curveName = dropType+" " + str(groupNum)
            y = getErrsOfDiffMethodsWithDiffAverage("fullConnected", "relu", "MNIST",
                                                  modelWidth, dropType, groupNum, keepProb, batchSize, perAverageEpochList)
            print(curveName + ":" + str(y))
            plt.plot(perAverageEpochList, y, color=colorMap[groupNum], label=curveName, linewidth=LINE_WIDTH,linestyle=lineStyleMap[dropType])

    plt.ylim(0.01, 0.05)
    plt.xticks(perAverageEpochList)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("epoch", fontsize=FONT_SIZE)
    plt.title("model width " + str(modelWidth), fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()


from test import getDataSaveDir
import os

def getConvergeData( modelWidth, dropType, groupNum, keepProb, batchSize, perAverageEpoch):
    if groupNum==1:
        perAverageEpoch=None

    if dropType == "singleOut":
        keepProb=None

    logDir = getDataSaveDir("MNIST", modelWidth, dropType, groupNum, keepProb, perAverageEpoch, batchSize)
    logFilePath=None

    for fileName in os.listdir(logDir):
        if "txt" in fileName:
            logFilePath=os.path.join(logDir,fileName)

    if logFilePath is None:
        raise Exception("no data in "+logDir)

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
            curveName = dropType+" " + str(groupNum)
            x,y = getConvergeData( modelWidth, dropType, groupNum, keepProb, batchSize, perAverageEpoch)
            plt.plot(x, y, color=colorMap[groupNum], label=curveName, linewidth=LINE_WIDTH,linestyle=lineStyleMap[dropType])

    plt.ylim(0.01, 0.05)
    #plt.xticks()

    plt.ylabel("test error",fontsize=FONT_SIZE)
    plt.xlabel("train step",fontsize=FONT_SIZE)
    plt.title("model width "+str(modelWidth),fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()


def visualizeRGBImgResult():
    dropoutTypeList = ["probOut", "singleOut"]
    datasetList=["CIFAR_10","SVHN"]

    modelWidthList = [0.25,0.5,1]

    groupNumList = [1, 2, 4]
    keepProb = 0.5
    batchSize = 256
    perAverageEpoch = 20

    dropType = "singleOut"
    dataset="SVHN"
    for groupNum in groupNumList:
        curveName = "group size" + str(groupNum)
        y = getErrsOfDiffMethodsWithDiffWidth("CNN", "relu", dataset,
                                              modelWidthList, dropType, groupNum, keepProb, batchSize, perAverageEpoch)
        print(curveName + ":" + str(y))
        plt.plot(modelWidthList, y, color=colorMap[groupNum], label=curveName, linewidth=LINE_WIDTH)

    #plt.ylim(0.01, 0.025)
    plt.xticks(modelWidthList)

    plt.ylabel("test error", fontsize=FONT_SIZE)
    plt.xlabel("model width", fontsize=FONT_SIZE)
    plt.title(dropType, fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE)
    plt.show()

##########################################################################################################################################

if __name__ == "__main__":
    loadTestErr("./result/fix_full.csv")
    loadTestErr("./result/fix_weight_avg_test.csv")
    loadTestErr("./result/2018_09_01 11_40_43.csv")
    printLowestErr()
    print("\n" + "*" * 10)
    #visualizeDiffMethods()
    #visualizeWeightAverage()
    #visualizeCoverge()
    visualizeRGBImgResult()