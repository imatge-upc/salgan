# Work space directory
HOME_DIR = '/home/titan/Saeed/saliency-salgan-2017/'

# Path to SALICON raw data
pathToImages = '/home/titan/Saeed/saliency-salgan-2017/data/dermoFitImage'
pathToMaps = '/home/titan/Saeed/saliency-salgan-2017/data/dermoFitMasks'
pathToImagesNoAugment = '/home/titan/Saeed/saliency-salgan-2017/data/train_img_cross'
pathToMapsNoAugment = '/home/titan/Saeed/saliency-salgan-2017/data/train_mask_cross'
pathToFixationMaps = ''
# Path to processed data
pathOutputImages = '/home/titan/Saeed/saliency-salgan-2017/data/image320x240'
pathOutputMaps = '/home/titan/Saeed/saliency-salgan-2017/data/mask320x240'
pathToPickle = '/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/trainData.pickle'
TRAIN_DATA_DIR_CROSS = '/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/trainDataNoAugment.pickle'
VAL_DATA_DIR = '/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/validationData.pickle'
TEST_DATA_DIR = '/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/titan/Saeed/saliency-salgan-2017/models/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (320,240)

# Directory to keep snapshots
DIR_TO_SAVE = '../weights'
FIG_SAVE_DIR = '../figs'

#Path to test images
pathToTestImages = '/home/titan/Saeed/saliency-salgan-2017/images'

#Path to segmentation resulta
pathToResultMaps = '/home/titan/Saeed/saliency-salgan-2017/segmentation'
