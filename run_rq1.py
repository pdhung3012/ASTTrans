from paths import *
from util import *
import ast
import json
import shutil
import traceback
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from statistics import mean,median
from tree_sitter import Language, Parser
import pickle
from nltk import ngrams
import codecs


fopRQ1=fopRepFolder+'data/rq1/'
fopQueryToASTTransRep=fopRQ1+'query-to-ASTTransRep/'
fopQueryToCodeTokens=fopRQ1+'query-to-codeTokens/'
fopResultsRQ1=fopRepFolder+'results/rq1/'
createDirIfNotExist(fopResultsRQ1)
lstFopNMTs=[fopQueryToCodeTokens,fopQueryToASTTransRep]
lstDatasets=['tlcodesum', 'csn', 'funcom', 'pcsd']
lstStrCommands=[]
fopConfigFolder=fopRepFolder+'data/configurations/'
fpConfigFile=fopConfigFolder+'standard.pkl'
dictConfigurations=pickle.load(open(fpConfigFile,'rb'))
isCacheNGram=dictConfigurations['isCacheNGram']
kTopFrequenceNGram=dictConfigurations['kTopFrequenceNGram']

fpSummary=fopResultsRQ1+'summary.txt'
f1=open(fpSummary,'w')
f1.write('NMT_Model\tDataset\tCrystalBLEU-4\tMeteor\n')
f1.close()

for fopNMT in lstFopNMTs:
    arrFopNMTs=fopNMT.split('/')
    currentConfig=arrFopNMTs[len(arrFopNMTs)-2]
    for currentDataset in lstDatasets:
        fopDataFolder='{}/{}/'.format(fopNMT,currentDataset)
        # arrFopItems=fopItem.split('/')
        fpTrainTarget = fopDataFolder + 'tgt-train.txt'
        fpTrainTrivialPkl = fopDataFolder + 'train-trivial-ngram.pkl'
        fpTrainTrivialText = fopDataFolder + 'train-trivial-ngram.txt'
        fpCompareCsv = fopResultsRQ1 + 'scores_{}_{}.csv'.format(currentDataset,currentConfig)
        fpTestTarget = fopDataFolder + 'tgt-test.txt'
        fpTestId = fopDataFolder + 'id-test.txt'
        fpPredict = fopDataFolder + 'pred.txt'

        try:
            arrLinePreds=[]
            if os.path.exists(fpPredict):
                f1 = open(fpPredict, 'r')
                arrLinePreds = f1.read().strip().split('\n')
                f1.close()
            trivially_shared_ngrams = {}
            isAbleSharedNGrams=False
            if isCacheNGram:
                try:
                    trivially_shared_ngrams=pickle.load(open(fpTrainTrivialPkl,'rb'))
                    isAbleSharedNGrams=True
                except Exception as e:
                    traceback.print_exc()
            if not isAbleSharedNGrams:
                f1=codecs.open(fpTrainTarget,'r','utf-8', errors = 'ignore')
                arrTrainTargets=f1.read().strip().split('\n')
                f1.close()
                frequencyNGrams={}

                for j in range(0,len(arrTrainTargets)):
                    try:
                        lstItemSplitCode=arrTrainTargets[j].split()
                        for indexNGram in range(3, 4):
                            lstItemNGrams = ngrams(lstItemSplitCode, indexNGram)
                            for gr in lstItemNGrams:
                                if gr not in frequencyNGrams.keys():
                                    frequencyNGrams[gr] = 1
                                else:
                                    frequencyNGrams[gr] += 1
                    except Exception as e:
                        traceback.print_exc()

                lstFreqNGramKeys = list(frequencyNGrams)[:kTopFrequenceNGram]
                trivially_shared_ngrams = {}
                for item in lstFreqNGramKeys:
                    # strAdd=' '.join(list(item))
                    trivially_shared_ngrams[item] = frequencyNGrams[item]
                pickle.dump(trivially_shared_ngrams,open(fpTrainTrivialPkl,'wb'))
                lstStr500 = []
                for qk in trivially_shared_ngrams.keys():
                    lstStr500.append('{} {}'.format(qk, trivially_shared_ngrams[qk]))
                f1 = open(fpTrainTrivialText, 'w')
                f1.write('\n'.join(lstStr500))
                f1.close()

            f1=open(fpTestTarget,'r')
            arrLineExps=f1.read().strip().split('\n')
            f1.close()
            f1=open(fpTestId,'r')
            arrLineIds=f1.read().strip().split('\n')
            f1.close()
            minLen=min([len(arrLineExps),len(arrLinePreds)])
            dictAllLinesCompared={}
            lstStrCompareCSV = ['Key\tCrystalBLEU\tMeteor']
            f1 = open(fpCompareCsv,'w')
            f1.write('\n'.join(lstStrCompareCSV)+'\n')
            f1.close()
            lstStrCompareCSV=[]
            print('{} {}'.format(currentDataset, fopNMT))
            for idx2 in range(0,minLen):
                try:
                    itemId=arrLineIds[idx2]
                    itemPred=arrLinePreds[idx2]
                    itemExp=arrLineExps[idx2]
                    lstStrPreds=itemPred.split()
                    lstStrExps=itemExp.split()
                    dictScoreJ = getSimilarityScoreCrystalBLEUAndMeteor(itemExp, itemPred, lstStrExps,
                                                    lstStrPreds, trivially_shared_ngrams)
                    dictScoreJ[itemId]=dictScoreJ
                    strLineAdd='{}\t{}\t{}'.format(itemId,dictScoreJ['c_b'],dictScoreJ['m'])
                    lstStrCompareCSV.append(strLineAdd)
                    if len(lstStrCompareCSV)%500==0 or idx2+1==minLen:
                        f1=open(fpCompareCsv,'a')
                        f1.write('\n'.join(lstStrCompareCSV)+'\n')
                        f1.close()
                except Exception as e:
                    traceback.print_exc()
            dfAllMetrics=pd.read_csv(fpCompareCsv,delimiter='\t')
            meanCryBLEU=dfAllMetrics['CrystalBLEU'].mean()
            meanMeteor=dfAllMetrics['Meteor'].mean()
            strLineAdd='{}\t{}\t{}\t{}'.format(currentConfig,currentDataset,meanCryBLEU,meanMeteor)
            f1=open(fpSummary,'a')
            f1.write(strLineAdd+'\n')
            f1.close()
            print('end folder  {} {}'.format(currentConfig,currentDataset))

        except Exception as e:
            traceback.print_exc()
