import numpy as np
import sys, os, traceback
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statistics import mean
import traceback
import numpy as np
from nltk.translate.bleu_score import sentence_bleu as originBleu
from bleu_ignoring import sentence_bleu  as crystalBLEU, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def getSimilarityScoreCrystalBLEUAndMeteor(str1,str2,lstStr1,lstStr2,trivially_shared_ngrams):
    dictItem={}
    # trivially_shared_ngrams=dictConfig['trivially_shared_ngrams']
    # rouge=dictConfig['rouge']
    try:
        crystalBLEU_score = crystalBLEU(
            [lstStr1], lstStr2, ignoring=trivially_shared_ngrams)
        meteor_sc = meteor_score([lstStr1], lstStr2)
        dictItem['c_b']=crystalBLEU_score
        dictItem['m']=meteor_sc
    except Exception as e:
        traceback.print_exc()
    return dictItem

def adjustScoreForMatrix(scores):
    try:
        minValue=np.min(scores)
        maxValue=np.max(scores)
        distance=maxValue-minValue
        scores=(scores-minValue)/distance
    except Exception as e:
        traceback.print_exc()
    return scores

def getReductionEmb(nl_vecs,code_vecs,reductType,reductSize):
    lenNLVecs=len(nl_vecs)
    lenCodeVecs=len(code_vecs)
    nl_vecs_transform=[]
    code_vecs_transform=[]
    all_vecs=nl_vecs+code_vecs
    # print('len all {}'.format(len(all_vecs)))
    if reductType=='adhoc':
        all_vecs_transform=[element[:reductSize] for element in all_vecs]
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]
    elif reductType=='pca':
        pca = PCA(n_components=reductSize)
        all_vecs_transform =pca.fit_transform(all_vecs)
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]
    else:
        tsne = TSNE(n_components=reductSize,
                    perplexity=40,
                    random_state=42,
                    n_iter=5000,
                    n_jobs=-1)
        all_vecs_transform = tsne.fit_transform(all_vecs)
        nl_vecs_transform = all_vecs_transform[:lenNLVecs]
        code_vecs_transform = all_vecs_transform[lenNLVecs:]

    return nl_vecs_transform.tolist(),code_vecs_transform.tolist()


def createDirIfNotExist(fopOutput):
    try:
        # Create target Directory
        os.makedirs(fopOutput, exist_ok=True)
        #print("Directory ", fopOutput, " Created ")
    except FileExistsError:
        print("Directory ", fopOutput, " already exists")

def exportDictToExcel(fpFile,lstRanks,dictInput):
    strHeader='Key,Rank,'
    lstTop1000=['Pos{}'.format(i) for i in range(1,1001)]
    strHeader=strHeader+','.join(lstTop1000)
    lstAllStrs=[strHeader]
    index=-1
    for key in dictInput.keys():
        index+=1
        val=dictInput[key]
        strLine='{},{},{}'.format(key,lstRanks[index],','.join(map(str,val)))
        lstAllStrs.append(strLine)
    f1=open(fpFile,'w')
    f1.write('\n'.join(lstAllStrs))
    f1.close()

def csm(A,B):
    num=np.dot(A,B.T)/(np.sqrt(np.sum(A**2,axis=1)[:,np.newaxis])*np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:])
    return num
