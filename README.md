# ASTTrans

## Prerequisites
Libraries: python 3.7 and above, meteor, crystalbleu (can download from https://github.com/sola-st/crystalbleu)

## Download
Download and extract the ASTTrans_RepPackages.zip from this following location: https://tinyurl.com/r5j8x5jc
In this folder, we have the "data" folder which stored information about translated/expected results of Query-To-ASTTrans Representation, Query-to-CodeTokens translation models, original vectors (by GraphCodeBERT and UniXcoder) and augmented vectors (by ASTTrans) over four datasets TLCodeSum, CodeSearchNet, Funcom and PCSD of CAT benchmark. The "result_backup" folder stored the results we used in the paper. When you run the reproducing code below, experiments' results will be shown in the "results" folder.


## Reproducing Paper Results
Change the default value of "fopRepPackage" variable in "src/paths.py" to your local location of "ASTTrans_RepPackages" folder. Then do following steps:
```bash
# Run RQ1: results are shown in "results/rq1" folder. See "summary.txt" to see results of Table 2.
python run_rq1.py

# Prepare for RQ2, RQ3: run code search by SOTA models (which returns the MRR for 4 datasets and 4 configurations per datasets in "originalCS/summary.txt")
python run_SOTA.py

# Run RQ2 and get summary of MRR in "results/combinedCS_standard/summary.txt". When you substract the results from "combinedCS_standard/summary.txt" by the results from "originalCS/summary.txt", you have the reported EffectMRR shown in Table 3)
python run_rq2.py

# Run RQ3 part 1 (using concatenated embedding): Get the MRR in "results/combinedCS_concat/summary.txt"
python run_rq3_concat.py

# Run RQ3 part 2 (changing combinedWeight): Get the MRR in "results/combinedCS_weight_*/summary.txt"
python run_rq3_combinedWeight.py

# Run RQ3 part 3 (changing depth of ASTTrans Representation): Get the MRR in "results/combinedCS_depth_*/summary.txt"
python run_rq3_depth.py
```
## Case Study
We manually checked 52 cases of RQ2 to see when ASTTrans improve/ cannot improve SOTA code search models. We store them in "caseStudy/" folder. In total, there are 20 cases ASTTrans improved SOTA approaches, 32 cases it didn't (20 cases are due to NMT's low quality translation and 12 cases are due to the repetitive of tokens in ASTTrans Representation)

## Confirmation from SOTA approaches' authors
We have two questions to SOTA authors about the validity of our implementation of code search by SOTA embedding models and the reasonable of the accuracy on the CAT benchmark's datasets. The confirmation that our implementation is valid and the obtained results are reasonable can be seen in the "QuestionsAndAnswersFromSOTAAuthors.pdf" in the above location (https://tinyurl.com/r5j8x5jc ).
