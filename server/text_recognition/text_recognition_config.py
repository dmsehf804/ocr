#recognition config
SAVE_MODEL= 'text_recognition/TPS-ResNet-BiLSTM-Attn.pth'

#data processing
IMAGE_HEIGHT=32
IMAGE_WIDTH=100
#LABLE='0123456789abcdefghijklmnopqrstuvwxyz'
LABLE='0123456789abcdefghijklmnopqrstuvwxyz'

#model Architecture
TRANSFORMATION='TPS' #None|TPS
FEATUREEXTRACTION='ResNet' #VGG|RCNN|ResNet
SEQUENCEMODELING='BiLSTM' #None|BiLSTM
PREDICTION='Attn' #CTC|Attn
NUM_FIDUCIAL=20 #'number of fiducial points of TPS-STN'
INPUT_CHANNEL=1 #the number of input channel of Feature extractor
OUTPUT_CHANNEL=512 #the number of output channel of Feature extractor
HIDDEN_SIZE=256 #the size of the LSTM hidden state

#variable
num_class=0
exp_name=None

#test
IMAGE_FOLDER='figures/'