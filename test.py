import os
import time

import numpy as np
import tensorflow as tf

import dataset
import models
import utils

#############################################################
PARALLEL_RANK=8
GPU_MEMORY_USE=0.6

WEIGHT_DECAY=0
STARTLREANINGRATE=1e-3
VALIDATE_RATE=0.2
EPOCHSPERCHECK=10

TEST_TIMES=5
USE_AUTO_TUNER=False
IS_RETRAIN_ON_ALL_TRAINSET=True
##############################################################

def getCurTimeStr():
    timeStruct=time.localtime(time.time())
    return time.strftime("%Y_%m_%d %H_%M_%S",timeStruct)

def getSaveVariable():
    return [v for v in tf.global_variables() if "Adam" not in v.name]

def getL2Loss():
    saveVariable=getSaveVariable()
    filters=[v for v in saveVariable if "kernel" in v.name]
    return tf.add_n([tf.nn.l2_loss(v) for v in filters])*WEIGHT_DECAY

def checkDir(baseDir,dirList):
    dirPath = baseDir
    for dirName in dirList:
        dirPath=os.path.join(dirPath,dirName)
        if os.path.exists(dirPath)==False:
            try:
                os.mkdir(dirPath)     #parallel exce may acc race condition here
            except Exception as e:
                print(str(e))
    return dirPath

def getDataSaveDir(datasetName,modelWidth, activateFunName, dropoutType, groupNum, keepProb,perAverageEpoch,batchSize):
    return checkDir("./",["result",datasetName,
                          str(dropoutType)+"_"+str(activateFunName)+"_"+str(modelWidth)+"_"+str(groupNum)+"_"+str(keepProb)+"_"+str(perAverageEpoch)+"_"+str(batchSize)])

def getModelSavePath(datasetName,modelWidth, activateFunName, dropoutType, groupNum, keepProb,perAverageEpoch,batchSize):
    baseDir=getDataSaveDir(datasetName,modelWidth, activateFunName, dropoutType, groupNum, keepProb,perAverageEpoch,batchSize)
    modelDir=checkDir(baseDir,["model"])
    return os.path.join(modelDir,"model")

#######################################################

def isNeedAugment(datasetName):
    if datasetName=="MNIST":
        return False
    return True

def getModelFun(modelFunName):
    if modelFunName=="fullConnected":
        return models.getFullConnectedModel
    if modelFunName=="CNN":
        return models.getSmallRGBImageModel
    raise Exception("unkonwn model fun:"+modelFunName)

def isNeedFlip(datasetName):
    if datasetName=="CIFAR_10":
        return True
    return False

def getActivateFun(activateFunName):
    if activateFunName=="relu":
        return tf.nn.relu
    elif activateFunName=="sigmoid":
        return tf.nn.sigmoid
    elif activateFunName=="tanh":
        return tf.nn.tanh

    raise Exception("unkown activation function:"+str(activateFunName))

def computErr(session, dataItertor, tensorMap):
    dataCount=0
    accCount=0
    for images,labels in dataItertor.getNextBatch():
        predictLabel=session.run([tensorMap["predictions"]],
                                 feed_dict={tensorMap["imagePlaceholder"]:images,
                                           tensorMap["isTrainPlaceholder"]:False})[0]
        dataCount+=len(images)
        accCount+=sum((predictLabel==labels).astype(np.int32))
    return 1-((accCount*1.0)/dataCount)

def trainModel(session, trainDataItertor, validateDataIterator, tuner, tensorMap, perAverageEpoch,modelPath,logDir,testDataIterator=None):
    step = 0
    validateErr=0.01
    curLearningRate = tuner.getLearningRate()

    if perAverageEpoch!=None:
        perAverageStep = int(perAverageEpoch * trainDataItertor.getDatasetSize() / trainDataItertor.getBatchSize())
    else:
        perAverageStep=None

    loss=tensorMap["loss"]
    accuracy=tensorMap["accuracy"]
    merged=tensorMap["merged"]
    trainOp=tensorMap["trainOp"]
    learningRatePlaceholder=tensorMap["learningRatePlaceholder"]
    imagePlaceholder=tensorMap["imagePlaceholder"]
    labelPlaceholder=tensorMap["labelPlaceholder"]
    isTrainPlaceholder=tensorMap["isTrainPlaceholder"]
    train_writer=tensorMap["train_writer"]
    averageOps=tensorMap["averageOps"]
    combineOps=tensorMap["combineOps"]
    saver=tensorMap["saver"]
    validateErrPlaceholder=tensorMap["validateErrPlaceholder"]

    logFile = None
    isSaved=False
    if testDataIterator!=None:
        logFile=open(os.path.join(logDir, getCurTimeStr()+".txt"), "w")

    while True:
        for images, labels in trainDataItertor.getNextBatch():
            if perAverageStep!=None and step % perAverageStep == 0:
                session.run(averageOps)

            r = session.run([loss, accuracy, merged, trainOp],
                            feed_dict={learningRatePlaceholder: curLearningRate,
                                       imagePlaceholder: images,
                                       labelPlaceholder: labels,
                                       isTrainPlaceholder: True,
                                       validateErrPlaceholder:validateErr})
            _loss = r[0]
            _accuracy = r[1]
            _merged = r[2]

            if step % 100 == 0:
                # print("step:"+str(step)+"  loss:"+str(_loss)+"  accuracy:"+str(_accuracy))
                train_writer.add_summary(_merged, step)

            if step%1000==0 and logFile is not None:
                session.run(combineOps)
                testErr = computErr(session, testDataIterator, tensorMap)
                logFile.write(str(step) + " " + str(testErr) + "\n")

            step += 1

        session.run(combineOps)
        validateErr = computErr(session, validateDataIterator, tensorMap)
        tuner.updateValidateErr(validateErr)
        if tuner.isShouldSave():
            saver.save(session, modelPath)  # used for early stop
            isSaved=True
        curLearningRate = tuner.getLearningRate()

        if tuner.isShouldStop():
            if isSaved==False:
                saver.save(session, modelPath)
            if logFile is not None:
                logFile.close()
            return

def testParam(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch):
    ###########################################
    #load data
    logDir=getDataSaveDir(datasetName, modelWidth, activateFunName, dropoutType, groupNum, keepProb,perAverageEpoch,batchSize)
    all_train_data, all_train_labels, test_data, test_labels=dataset.getDataset(datasetName, "./datasets")

    all_train_data=all_train_data.astype(np.float32)
    test_data=test_data.astype(np.float32)

    temp_train_data, temp_train_labels=utils.shuffData(all_train_data, all_train_labels) #return copyed data
    validateDataLen=int(len(temp_train_data)*VALIDATE_RATE)
    validate_data=temp_train_data[:validateDataLen,:,:,:]
    validate_labels=temp_train_labels[:validateDataLen]
    part_train_data=temp_train_data[validateDataLen:,:,:,:]
    part_train_labels=temp_train_labels[validateDataLen:]

    utils.normalizeData(all_train_data)
    utils.normalizeData(part_train_data)
    utils.normalizeData(validate_data)
    utils.normalizeData(test_data)

    if isNeedAugment(datasetName):
        allTrainDataIterator = utils.AugmentDatasetLiterator(all_train_data, all_train_labels, EPOCHSPERCHECK, batchSize,isNeedFlip(datasetName))
        partTrainDataIterator = utils.AugmentDatasetLiterator(part_train_data, part_train_labels, EPOCHSPERCHECK, batchSize,isNeedFlip(datasetName))
    else:
        allTrainDataIterator = utils.DatasetLiterator(all_train_data, all_train_labels, EPOCHSPERCHECK,batchSize)
        partTrainDataIterator = utils.DatasetLiterator(part_train_data, part_train_labels, EPOCHSPERCHECK,batchSize)
    validateDataIterator = utils.DatasetLiterator(validate_data, validate_labels, 1, batchSize)
    testDataIterator = utils.DatasetLiterator(test_data, test_labels, 1, batchSize)

    ################################################
    #build model
    imageShape=list(temp_train_data.shape)
    imageShape[0]=None
    imagePlaceholder=tf.placeholder(tf.float32,imageShape)
    labelPlaceholder=tf.placeholder(tf.int32,[None])
    isTrainPlaceholder=tf.placeholder(tf.bool)
    learningRatePlaceholder = tf.placeholder(tf.float32, name="learningRate")
    validateErrPlaceholder = tf.placeholder(tf.float32)

    modelFun=getModelFun(modelFunName)
    logits,combineOps, averageOps=modelFun(imagePlaceholder, modelWidth, dataset.getDatasetClassNum(datasetName),
                                           getActivateFun(activateFunName), dropoutType, groupNum, keepProb, isTrainPlaceholder)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labelPlaceholder, logits=logits)
    predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labelPlaceholder), tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRatePlaceholder)
    trainOp = optimizer.minimize(loss)
    saver = tf.train.Saver(getSaveVariable())

    tf.summary.scalar("1_accurate_", accuracy)
    tf.summary.scalar("2_loss_", loss)
    tf.summary.scalar("3_learningRate_", learningRatePlaceholder)
    tf.summary.scalar("4_validateErr_",validateErrPlaceholder)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logDir, graph=tf.get_default_graph())

    tensorMap = {"predictions": predictions,
                 "isTrainPlaceholder": isTrainPlaceholder,
                 "imagePlaceholder": imagePlaceholder,
                 "labelPlaceholder": labelPlaceholder,
                 "loss":loss,
                 "accuracy":accuracy,
                 "merged":merged,
                 "trainOp":trainOp,
                 "learningRatePlaceholder":learningRatePlaceholder,
                 "train_writer":train_writer,
                 "combineOps":combineOps,
                 "averageOps":averageOps,
                 "saver":saver,
                 "validateErrPlaceholder":validateErrPlaceholder}

    ####################################################
    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=(GPU_MEMORY_USE)/(PARALLEL_RANK)
    session = tf.Session(config=config)
    session.run([tf.global_variables_initializer()])

    modelPath=getModelSavePath(datasetName,modelWidth, activateFunName, dropoutType, groupNum, keepProb,perAverageEpoch,batchSize)
    errs=[]
    supConfigs=[]

    #globalAutoTuner = utils.GlobalLearningRateTuner(STARTLREANINGRATE)
    localAutoTuner=utils.LocalLearningRateTuner(STARTLREANINGRATE,maxDontSave=7)
    autoTuners=[localAutoTuner]

    #find train param base on validate data
    if USE_AUTO_TUNER:
        for autoTuner in autoTuners:
            session.run([tf.global_variables_initializer()])
            trainModel(session, partTrainDataIterator, validateDataIterator, autoTuner, tensorMap, perAverageEpoch,
                    modelPath, logDir)
            saver.restore(session, modelPath)  # load early stop model
            errs.append(computErr(session, testDataIterator, tensorMap))

            if IS_RETRAIN_ON_ALL_TRAINSET:
                supConfigs.append((autoTuner.getFixTuner(),allTrainDataIterator,testDataIterator))
    else:
        fixTuner1 = utils.getFixLearningRateTuner([10, 10], [1e-3, 1e-4], isEarlyStop=False)
        supConfigs.append((fixTuner1, allTrainDataIterator, testDataIterator))

    for tuner,trainIterator,validateIterator in supConfigs:
        session.run([tf.global_variables_initializer()])
        trainModel(session, trainIterator, validateIterator, tuner, tensorMap, perAverageEpoch, modelPath, logDir)
        saver.restore(session,modelPath)    #load model
        errs.append(computErr(session, testDataIterator, tensorMap))

    session.close()

    if len(errs)==0:
        raise Exception("config error")

    return [min(errs)]

#########################################################
def describeArgs(args):
    if type(args)==list:
        des=""
        for i in range(len(args)):
            if i<len(args)-1:
                des+=str(args[i])+","
            else:
                des += str(args[i])
        return des
    return str(args)

def computeParamCombination(modelFunName, activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch):
    #all param should be list
    paramCombinationList=[[f] for f in modelFunName]
    for params in [activateFunName, datasetName, modelWidth, dropoutType, groupNum, keepProb, batchSize,perAverageEpoch]:
        tempCombinationList=[]

        for combination in paramCombinationList:
            if ((params == keepProb) and ("singleOut" in combination)) or \
                    ((params == perAverageEpoch) and (combination[-3]==1)):
                tempCombination = combination[:]
                tempCombination.append(None)
                tempCombinationList.append(tempCombination)
            else:
                for param in params:
                    tempCombination=combination[:]
                    tempCombination.append(param)
                    tempCombinationList.append(tempCombination)

        paramCombinationList=tempCombinationList
    return paramCombinationList

def getMNISTParam():
    activateFunctions=["relu","sigmoid","tanh"]
    datasetName = ["MNIST"]
    modelWidth = [256]
    dropoutType = ["probOut", "singleOut"]
    groupNum = [1,2,4]
    keepProb = [0.5]
    batchSize = [256]
    perAverageEpoch=[10]

    return computeParamCombination(["fullConnected"],activateFunctions,
                                   datasetName,modelWidth, dropoutType, groupNum, keepProb,batchSize,perAverageEpoch)

def getRGBImageDatasetParam():
    datasetName=["CIFAR_10"]
    modelWidth=[0.25,0.5,1]
    dropoutType=["singleOut"]
    groupNum=[1,2,4]
    keepProb=[0.5]
    batchSize=[256]
    perAverageEpoch=[10]

    return computeParamCombination(["CNN"],["relu"],
                                   datasetName,modelWidth, dropoutType, groupNum, keepProb,batchSize,perAverageEpoch)

#########################################################

from multiprocessing import Process,Queue
import threading

def poxyFun(argsWrap):
    msgQueue, args=argsWrap

    errs=testParam(*args)
    msgQueue.put(errs)

def testInOtherProcess(args):
    msgQueue = Queue()
    argsWrap=[msgQueue, args]

    p = Process(target=poxyFun, args=[argsWrap])
    p.start()
    p.join()

    return msgQueue.get()

def workThread(getTaskFun,returnResultFun):
    while True:
        args=getTaskFun()
        if args == None:
            return

        try:
            for _ in range(TEST_TIMES):
                errs=testInOtherProcess(args)
                returnResultFun(args,errs)
                time.sleep(10)
        except Exception as e:
            print(str(e))

def main():
    allArgsList = getMNISTParam()
    #allArgsList = getRGBImageDatasetParam()
    resultMap = {}

    print("\ntest param:")
    for args in allArgsList:
        print(args)
        resultMap[describeArgs(args)] = []
    print("\n"*2)

    time.sleep(5) #for check experiment setting

    taskLock=threading.Lock()
    resultLock=threading.Lock()

    def getTaskFun():
        args=None
        taskLock.acquire()
        if len(allArgsList)!=0:
            args=allArgsList.pop(0)
        taskLock.release()
        return args

    def returnResultFun(args,errs):
        resultLock.acquire()
        resultMap[describeArgs(args)] += errs
        resultLock.release()
        print(describeArgs(args)+","+str(errs))

    threadList=[]
    for _ in range(PARALLEL_RANK):
        t=threading.Thread(target=workThread,args=[getTaskFun,returnResultFun])
        time.sleep(10)
        t.start()
        threadList.append(t)

    for t in threadList:
        t.join()

    reportList=[]
    print("\n" * 3 + "*" * 20)
    for key in list(resultMap.keys()):
        errs=resultMap[key]
        report = key + "," + str(sum(errs)/len(errs)) + "," + str(errs)
        print(report)
        reportList.append(report)
    with open("./result/"+getCurTimeStr()+".csv","w") as f:
        for r in reportList:
            f.write(r+"\n")

############################################################

if __name__=="__main__":
    main()
