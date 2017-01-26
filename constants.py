# Work space directory
HOME_DIR = '/home/users/jpang/scratch-local/lsun2016/salgan-cvpr-2017/'

# Path to SALICON raw data
pathToImages = '/home/users/jpang/salicon_data/images'
pathToMaps = '/home/users/jpang/salicon_data/saliency'
pathToFixationMaps = '/home/users/jpang/salicon_data/fixation'

# Path to processed data
pathOutputImages = '/home/users/jpang/lsun2016/data/salicon/images320x240'
pathOutputMaps = '/home/users/jpang/lsun2016/data/salicon/saliency320x240'
pathToPickle = '/home/users/jpang/scratch-local/salicon_data/320x240'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/home/users/jpang/scratch-local/salicon_data/320x240/trainData.pickle'
VAL_DATA_DIR = '/home/users/jpang/scratch-local/salicon_data/320x240/fix_validationData.pickle'
TEST_DATA_DIR = '/home/users/jpang/scratch-local/salicon_data/256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/scratch/local/jpang/vgg16.pkl'
PATH_TO_RESNET50_WEIGHTS = '/scratch/local/jpang/resnet50.pkl'
PATH_TO_FCN_WEIGHTS = '/home/users/jpang/scratch-local/ConvVGG_FCN.pkl'

# Input image and saliency map size
INPUT_SIZE = (320, 240)

# Directory to keep snapshots
# DIR_TO_SAVE = 'test'
# DIR_TO_SAVE = 'resnet50'
# DIR_TO_SAVE = 'resnet50_gan'
DIR_TO_SAVE = 'fcn_gan'