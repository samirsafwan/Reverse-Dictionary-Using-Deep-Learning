MODEL_LSTM = "lstm"
MODEL_BASELINE = "avg"

METRIC_NORM = "norm"
METRIC_DOTPROD = "dot"
METRIC_COSINE = "cos"

ATTENTION = True

EPSILON = 1e-6

################# DEFAULT SELECTION #####################
EMBEDDING_DIM = 50
NUMRANKS = 10
METRIC_TYPE = METRIC_NORM
MODEL_TYPE = MODEL_BASELINE
METRIC_THRESHOLD = 0.5

################# LSTM params #####################
INPUT_SIZE    = EMBEDDING_DIM
HIDDEN_LAYERS = 20
OUTPUT_SIZE   = EMBEDDING_DIM
LEARNING_RATE = 0.001
MAX_PHRASE_LEN= 10
ITERATIONS_PER_EPOCH = 10
BATCH_SIZE = 500
EPOCHS = 10
REG_CONST = 0.0001

################# PATHS ##########################
TEST_DATA_PATH = 'data/test.txt'
MANUAL_DATA_PATH = 'data/webstertest.txt'
TRAINED_MODEL_PATH = 'model/trained_model'
GLOVE_FILEPATH = "data/glove.6B."+str(EMBEDDING_DIM)+"d.txt"
