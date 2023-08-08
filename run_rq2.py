from util import *
import glob
import pickle
import numpy as np
import time
import traceback




from paths import *
fopConfigFolder=fopRepFolder+'data/configurations/'
fopVectorsOriginal=fopRepFolder+'data/vectors_original/'
fopVectorsAugmented=fopRepFolder+'data/rq2/vectors_augmented/'

fopResultCombine=fopRepFolder+'results/combineCS_standard/'
createDirIfNotExist(fopResultCombine)
fpConfigFile=fopConfigFolder+'standard.pkl'
dictConfigurations=pickle.load(open(fpConfigFile,'rb'))
topSelect=dictConfigurations['topSelect']

lstProjectNames=dictConfigurations['lstProjectNames']
augmentedSize=dictConfigurations['augmentedSize']
reductType=dictConfigurations['reductType']
dictWAsAllProjects=dictConfigurations['dictWA']
lstOrgEmbModels=dictConfigurations['lstOrgEmbModels']
lstOrgDimensionSize=dictConfigurations['lstSizeOfReduction']
cacheSizeForVector=dictConfigurations['cacheSizeForVector']
fpSummary = fopResultCombine + 'summary.txt'
f1 = open(fpSummary, 'w')
f1.write('Dataset\tOriginal Model\tOrg. Emb. Dim. Size\tMRR\n')
f1.close()

for idxProjectName in range(0,len(lstProjectNames)):
    currentProjectName=lstProjectNames[idxProjectName]
    dictWeightAugmented=dictWAsAllProjects[currentProjectName]

    fopDetailsRank = fopResultCombine + '{}/details_rank/'.format(currentProjectName)
    fopDetailsTop = fopResultCombine + '{}/details_top/'.format(currentProjectName)
    fopPredictedRank = fopResultCombine + '{}/predicted_rank/'.format(currentProjectName)
    fopStoreSortedIndex = fopResultCombine + '{}/store_sorted_ids/'.format(currentProjectName)
    createDirIfNotExist(fopStoreSortedIndex)
    createDirIfNotExist(fopDetailsRank)
    createDirIfNotExist(fopDetailsTop)
    createDirIfNotExist(fopPredictedRank)

    fpAugmentedVectors = '{}/{}.test.pkl'.format(fopVectorsAugmented, currentProjectName)
    dictTestEmb = pickle.load(open(fpAugmentedVectors, 'rb'))
    setTestKeys = set(list(dictTestEmb.keys()))

    for embeddingModel in lstOrgEmbModels:
        fpOriginalVectors= '{}/{}/{}.test.pkl'.format(fopVectorsOriginal,embeddingModel,currentProjectName)
        for reductSize in lstOrgDimensionSize:
            try:
                strItemId = '{}__{}__{}'.format(currentProjectName, embeddingModel, reductSize)
                print('begin {}'.format(strItemId))
                dictQueriesCands = pickle.load(open(fpOriginalVectors, 'rb'))
                dictAllQueries = dictQueriesCands['queries']
                dictAllCands = dictQueriesCands['candidates']
                dictNewIdQueries = {}
                dictNewIdCands = {}
                for key in dictAllQueries.keys():
                    idQuery = int(key.split('_')[1])
                    if str(idQuery) in setTestKeys:
                        dictNewIdQueries[idQuery] = dictAllQueries[key]
                for key in dictAllCands.keys():
                    idCand = int(key.split('_')[1])
                    if str(idCand) in setTestKeys:
                        dictNewIdCands[idCand] = dictAllCands[key]

                dictNewIdQueries = dict(sorted(dictNewIdQueries.items()))
                dictNewIdCands = dict(sorted(dictNewIdCands.items()))
                listKeyQueries = list(dictNewIdQueries.keys())[:topSelect]
                listKeyCands = list(dictNewIdCands.keys())[:topSelect]
                listVectorQueries = list(dictNewIdQueries.values())[:topSelect]
                listVectorCands = list(dictNewIdCands.values())[:topSelect]
                listVectorQueries = [a.tolist() for a in listVectorQueries]
                listVectorCands = [a.tolist() for a in listVectorCands]

                if len(listVectorQueries[0])>reductSize:
                    nl_vecs, code_vecs = getReductionEmb(listVectorQueries, listVectorCands, reductType, reductSize)
                else:
                    nl_vecs, code_vecs =listVectorQueries,listVectorCands

                nlAug_vecs = []
                codeAug_vecs = []
                for indexKey in range(0, len(listKeyQueries)):
                    keyQuery = '{}'.format(listKeyQueries[indexKey])
                    # print(dictTestEmb.keys())
                    # input('bbbb')
                    if keyQuery in dictTestEmb.keys():
                        valAug = dictTestEmb[keyQuery]
                        lstAugExp = [it for it in valAug['exp']]
                        lstAugPred = [it for it in valAug['pred']]
                        nlAug_vecs.append(lstAugPred)
                        codeAug_vecs.append(lstAugExp)

                    else:
                        itemAugQuery = [0 for i in range(0, augmentedSize)]
                        itemAugCand = [0 for i in range(0, augmentedSize)]
                        nlAug_vecs.append(lstAugPred)
                        codeAug_vecs.append(lstAugExp)
                        print('go here')

                nl_vecs = [[a] for a in nl_vecs]
                code_vecs = [[a] for a in code_vecs]
                code_vecs = np.concatenate(code_vecs, 0)
                nl_vecs = np.concatenate(nl_vecs, 0)

                nlAug_vecs = [[a] for a in nlAug_vecs]
                codeAug_vecs = [[a] for a in codeAug_vecs]
                codeAug_vecs = np.concatenate(codeAug_vecs, 0)
                nlAug_vecs = np.concatenate(nlAug_vecs, 0)

                # print('{} {} {}'.format(len(listVectorCands),type(listVectorCands[0]),listVectorCands[0]))
                start_time = time.time()
                strKeyConfig='{}_{}'.format(embeddingModel,reductSize)
                if strKeyConfig in dictWeightAugmented.keys():
                    combineWeight=dictWeightAugmented[strKeyConfig]
                    # print('com {}'.format(combineWeight))
                # combineWeight=0
                scores = csm(nl_vecs, code_vecs)
                scores = adjustScoreForMatrix(scores)
                scoresAug = csm(nlAug_vecs, codeAug_vecs)
                scores = scores * (1 - combineWeight) + scoresAug * combineWeight
                sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
                del scores, scoresAug

                ranks = []
                realRanks = []
                lstPredictedTop1s = []
                # dictDetailResults={}
                # zipData = zip(listKeyQueries, sort_ids)
                currentIndexSortedIndex = -1
                dictBatchIndexSorted = {}
                # for url, sort_id in zipData:
                for indexScore in range(0, len(listKeyQueries)):
                    url = listKeyQueries[indexScore]
                    newIndexSortedIndex = indexScore // cacheSizeForVector
                    if newIndexSortedIndex != currentIndexSortedIndex:
                        fpBatchSortedIndex = fopStoreSortedIndex + '{}_{}_{}.pkl'.format(reductSize,embeddingModel,currentIndexSortedIndex)
                        pickle.dump(dictBatchIndexSorted, open(fpBatchSortedIndex, 'wb'))
                        dictBatchIndexSorted = {}

                    sort_id = sort_ids[indexScore]
                    dictBatchIndexSorted[indexScore] = sort_id
                    rank = 0
                    find = False
                    # lstTop1000PerKey=[]
                    lstPredictedTop1s.append(listKeyCands[sort_id[0]])
                    for idx in sort_id[:1000]:
                        # lstTop1000PerKey.append(listKeyCands[idx])
                        if find is False:
                            rank += 1
                        if listKeyCands[idx] == url:
                            find = True
                    # dictDetailResults[url]=lstTop1000PerKey
                    if find:
                        realRanks.append(rank)
                        ranks.append(1 / rank)

                    else:
                        ranks.append(0)
                        realRanks.append(1001)
                    if (indexScore + 1) == len(listKeyQueries):
                        fpBatchSortedIndex = fopStoreSortedIndex + '{}_{}_{}.pkl'.format(reductSize,
                                                                                                       embeddingModel,
                                                                                                       currentIndexSortedIndex)
                        pickle.dump(dictBatchIndexSorted, open(fpBatchSortedIndex, 'wb'))
                        dictBatchIndexSorted = {}
                del sort_id, find, rank


                mrrScore = np.mean(ranks)
                duration = (time.time() - start_time)
                lstKeyAndRank = ['{},{}'.format(listKeyQueries[i], realRanks[i]) for i in
                                 range(0, len(listKeyQueries))]
                lstKeyAndRank=['QueryId,Rank']+lstKeyAndRank
                strItemId = '{}__{}__{}'.format(currentProjectName,embeddingModel, reductSize)
                f1 = open(fopDetailsRank + strItemId+'.csv', 'w')
                f1.write('\n'.join(lstKeyAndRank))
                f1.close()

                lstPredictAndRank = ['{},{}'.format(listKeyQueries[i], lstPredictedTop1s[i]) for i in
                                     range(0, len(listKeyQueries))]
                lstPredictAndRank = ['QueryId,PredictTop1Id'] + lstPredictAndRank
                strItemId = '{}__{}__{}'.format(currentProjectName, embeddingModel, reductSize)
                f1 = open(fopPredictedRank + strItemId + '.csv', 'w')
                f1.write('\n'.join(lstPredictAndRank))
                f1.close()
                strLineMRRScore = '{}\t{}\t{}\t{}'.format( currentProjectName, embeddingModel,reductSize,
                                                                  mrrScore)
                print(strLineMRRScore)
                f1 = open(fpSummary, 'a')
                f1.write(strLineMRRScore + '\n')
                f1.close()
                del lstKeyAndRank
                del lstPredictAndRank
                ranks = None
                realRanks = None
                nl_vecs = None
                code_vecs = None
                zipData = None
                # listKeyQueries=None
                scores = None
                sort_ids = None

            except Exception as e:
                traceback.print_exc()
