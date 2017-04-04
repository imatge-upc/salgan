import os
import numpy as np
import cv2
import sys
import glob
from tqdm import tqdm
from constants import *
from sklearn.metrics import jaccard_similarity_score
def evaluate():

    listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResMaps,'*.bmp'))]
    err = 0.
    jaccard_score=0.
    dice = 0.
    dice_err = 0.
    for currFile in tqdm(listImgFiles):

        #res = cv2.imread(os.path.join(pathToResMaps,currFile+'.bmp'),cv2.IMREAD_GRAYSCALE)
   	res = np.load(os.path.join(pathToResMaps,currFile+'.npy'))
# 	res = cv2.normalize(res.astype('float'),None,0.0,1.0,cv2.NORM_MINMAX)
	gt = np.float32(cv2.imread(os.path.join(pathToMaps,currFile+'mask.png'),cv2.IMREAD_GRAYSCALE))/255
	#print "Filename-->", currFile, " ResSize--> ", res.shape, "GtSize:--> ", gt.shape
   	#pixelwise_err = np.sum((res-gt)**2)
	jaccard_score += jaccard_similarity_coefficient(res,gt)
        #dice += np.sum(res[gt==1])*2.0 / (np.sum(res) + np.sum(gt))
	#pixelwise_err /= float(res.shape[0]*res.shape[1])
        #err += pixelwise_err

    #err /= len(listImgFiles)
    #dice /= len(listImgFiles)
    jaccard_score /= len(listImgFiles)
    #print "Error: ", err
    #print "Acc: ", 1-err
    print "Jaccard: ", jaccard_score
    #print "Dice: ", dice

def jaccard_similarity_coefficient(A, B, no_positives=1.0):
    """Returns the jaccard index/similarity coefficient between A and B.
    
    This should work for arrays of any dimensions.
    
    J = len(intersection(A,B)) / len(union(A,B)) 
    
    To extend to probabilistic input, to compute the intersection, use the min(A,B).
    To compute the union, use max(A,B).
    
    Assumes that a value of 1 indicates the positive values.  
    A value of 0 indicates the negative values.
    
    If no positive values (1) in either A or B, then returns no_positives.
    """
    # Make sure the shapes are the same.
    if not A.shape == B.shape:
        raise ValueError("A and B must be the same shape")
        
        
    # Make sure values are between 0 and 1.
    if np.any( (A>1.) | (A<0) | (B>1.) | (B<0)):
        raise ValueError("A and B must be between 0 and 1")
    
    # Flatten to handle nd arrays.
    A = A.flatten()
    B = B.flatten()
    
    intersect = np.minimum(A,B)
    union = np.maximum(A, B)
    
    # Special case if neither A or B have a 1 value.
    if union.sum() == 0:
        return no_positives
    
    # Compute the Jaccard.
    J = float(intersect.sum()) / union.sum()
    return J


def _jaccard_similarity_coefficient():
    A = np.asarray([0,0,1])
    B = np.asarray([0,1,1])
    exp_out = 1. / 2
    act_out = jaccard_similarity_coefficient(A,B)
    assert act_out == exp_out, "Incorrect jaccard calculation"

    A = np.asarray([0,1,1])
    B = np.asarray([0,1,1])
    act_out = jaccard_similarity_coefficient(A,B)
    assert act_out == 1., "If same, then is 1."

    A = np.asarray([0,1,1,0])
    B = np.asarray([0,1,1])
    try: excep = False; jaccard_similarity_coefficient(A,B) # This should throw an error.
    except: excep = True
    assert excep, "Error with different sized inputs."

    A = np.asarray([0,1,1.1])
    B = np.asarray([0,1,1])
    try: excep = False; jaccard_similarity_coefficient(A,B) # This should throw an error.
    except: excep = True
    assert excep, "Values should be between 0 and 1."

    A = np.asarray([1,0,1])
    B = np.asarray([0,1,1])
    A2 = np.asarray([A,A])
    B2 = np.asarray([B,B])
    act_out = jaccard_similarity_coefficient(A2,B2)
    assert act_out == 1. / (3), "Incorrect 2D jaccard calculation."

    # Fuzzy values.
    A = np.asarray([0.5,0,1])
    B = np.asarray([0.6,1,1])
    exp_out = 1.5 / (0.6+2)
    act_out = jaccard_similarity_coefficient(A,B)
    assert act_out == exp_out, "Incorrect fuzzy jaccard calculation."
    
#_jaccard_similarity_coefficient()

def main():
    evaluate()

if __name__ == '__main__':
    main()


