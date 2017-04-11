import os
import numpy as np
import cv2
import sys
import glob
from tqdm import tqdm
from constants import *
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix

def evaluate():
    listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToImages,'test*.png'))]
    jaccard_score=0.
    dice = 0.
    spec = 0.
    sens = 0.
    acc = 0.
    for currFile in tqdm(listImgFiles):
#        res = np.float32(cv2.imread(os.path.join(pathToResMaps,currFile+'.png'),cv2.IMREAD_GRAYSCALE))/255
   	res = np.load(os.path.join(pathToResMaps,currFile+'_gan.npy'))
	gt = np.float32(cv2.imread(os.path.join(pathToMaps,currFile+'mask.png'),cv2.IMREAD_GRAYSCALE))/255

	jaccard_score += jaccard_similarity_coefficient(gt,res)
        dice += dice_coefficient(gt, res)
      	spec_tmp, sens_tmp, acc_tmp = specificity_sensitivity(gt, res)
     	spec += spec_tmp
 	sens += sens_tmp
	acc += acc_tmp

    dice /= len(listImgFiles)
    jaccard_score /= len(listImgFiles)
    spec /= len(listImgFiles)
    sens /= len(listImgFiles)
    acc /= len(listImgFiles)
    
    print "Accuracy: ", acc
    print "Sensitivity: ", sens
    print "Specificity: ", spec
    print "Jaccard: ", jaccard_score
    print "Dice: ", dice

def dice_coefficient(gt, res):

    A = gt.flatten()
    B = res.flatten()
    
    A = np.array([1 if x > 0.5 else 0.0 for x in A])
    B = np.array([1 if x > 0.5 else 0.0 for x in B])
    dice = np.sum(B[A==1.0])*2.0 / (np.sum(B) + np.sum(A))
    return dice
def specificity_sensitivity(gt, res):
    A = gt.flatten()
    B = res.flatten()

    A = np.array([1 if x > 0.5 else 0.0 for x in A])
    B = np.array([1 if x > 0.5 else 0.0 for x in B])

    tn, fp, fn, tp = np.float32(confusion_matrix(A, B).ravel())
    specificity = tn/(fp + tn)
    sensitivity = tp / (tp + fn)
    accuracy = (tp+tn)/(tp+fp+fn+tn)

    return specificity, sensitivity,accuracy

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


