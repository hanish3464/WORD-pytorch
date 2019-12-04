##TEST CASE#
TEST_CASE = '너미디어타다보장전선멋다올글자관심무기남대문보완하다바발달하다주다자기짓다전과민반응이라고자내한을몰라아무리노력해도상대잡을수없을때의그괴감을저친구는말이제스물셋인데백여명의부하거느린보스로군림하고있최고의자리에있놈들은뭔가남들과다른게있을거그자리에안올라가본자는영원히모를뭔가가나는말이야태어날때부터사람의그릇이란정해져있다고믿네여깁니다형님이겁니까?너카타쿰의주인라플라스의둥지에추락했다나도아그래서?멍청한놈탁철민이바보라서그런제안을했겠어?뭔가있을게니야살점을몇개어주더라도뼈째뜯어먹을뭔하지만인도네시에는더이상대형프로젝트가없는걸로아는데라플라스의기원은다른귀족들에비하면턱없이짧다라플라스는영민이올것을이미1에알고있었다그기간동안그는온갖수단을다동원해힘을키다에테르가충만한이카타쿰에둥지를튼것도바로그중하나다신과싸우던거인족이라는전승이있는그트롤들을이렇게쉽게잡다그럼갑시다언젠가는이런일이있을줄은알고있었지만네?그러십니까?회장님께서는이일을빨리처리하길원하십니다카타쿰말입니까?멋굶다올글자관심거건물카타쿰어서현실로도망저사슬은뭐죠?그래서뭐?그여러분은이만나가보세요형현실의시흑흑가인박명이라더니기구한내팔무기턱이남자당신과함께있었던소년은누구죠?정말이에어때요?찾을수있겠습니까?에테르젬수완료무구제작미션목표치까지필요한에테르젬2623아직데전현억대토유형남성시설걱정발전하다저녁때업만나다업차탓운전발전되다서른확장옳다출구속도얼마나매'

#TRANSFER CASE#
TRANSFER_CASE = '사막락락다나날락았락마만가갑락완아라만라말짜정정건치치잠깐아싸럼만해만님선케나준남자정말너는과일는너아선물?!차만?안!!업나나자마자바산자잘가장잘알막라장갑자까다다만낙짜판잡ㅋㅋ하하ㅋㅋ한한다ㅋㅋ한다할ㅋ한나쌍왕따와말ㅋ과ㅋ맞각난ㅋㅋ말다ㅋ한만습하놔박습쏠짝1까님습말잘하한아라안빨아만화나라마가하하빨가'

TRAIN_IMAGE_PATH = './train/images/'
TEST_IMAGE_PATH = './test/images/'
TEST_CUSTOM_IMAGE_PATH = './test/custom/'
TEST_PREDICTION_PATH = './test/predictions/result-'

TRANFSER_TRAIN_IMAGE_PATH = './train/transfer_train/images/'
TRANSFER_TRAIN_CSV_PATH = './train/transfer_train/label.csv'

TRAIN_FONTS_PATH = './train/fonts/'
TEST_FONTS_PATH = './test/fonts/'

TRAIN_CSV_PATH = './train/label.csv'
TEST_CSV_PATH = './test/label.csv'
TEST_CUSTOM_CSV_PATH = 'test/custom/label.csv'
CUSTOM_TEST_GT_PATH = './test/custom/gt.txt'

SAVED_MODEL_PATH = './saved_models/'
PRETRAINED_MODEL_PATH = './pretrained_models/Webtoon-Text-Recognizer.pth'
LABEL_PATH = './train/labels-2198.txt'

# IMAGE GENERATION PARAMETER VALUE
IMAGE_WIDTH = IMAGE_HEIGHT = 64
FONT_SIZE = 48
BACKGROUND = 0
FONT_COLOR = 255
DISTORTION_TIMES = 4
ALPHA_MIN = 40
ALPHA_MAX = 45
SIGMA_MIN = 6
SIGMA_MAX = 7
BLUR_EXTENT = 2
ROTATION_ANGLE = 20
MORPH_NUM = 2
NOISE_GEN_NUM = 3
MODEL_CHANNEL = 1

# HYPERPAMETER AND CONFIGURATION
CUDA = True
MULTI_GPUS = False
LEARNING_RATE = 0.01
TRANSFER_LEARNING_RATE = 0.00001
TRANSFER_LR_DECAY_STEP = 200
LR_DECAY_STEP = 5
LR_DECAY_GAMMA = 0.1
BATCH = 64
TRANSFER_BATCH = 100
TEST_BATCH = 100
EPOCH = 20
TRANSFER_EPOCH = 5
TARGET_IMAGE_SIZE = 224 #224
NUM_CLASSES = 2198
DISPLAY_INTERVAL = 100

#PIPELINEING WITH TEXT DETECTION RESULT
TEST_RECOG_PATH = '../text_detection/test/predictions/res/'
SPACING_WORD_PATH = '../text_detection/test/predictions/spacing_word/spacing_word.txt'
