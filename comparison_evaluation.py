"""Hazard Zone."""

from fee.classification import Expression as Exp
from fee.data import FER2013Dataset
from keras.models import load_model
import numpy
import sys

# - Global Vars ---------------------------------------------------------------
# -----------------------------------------------------------------------------

if len(sys.argv) != 4:
    print("Gib FER2013 path, created model path and result target path pls.")
    exit()

fer2013_path = sys.argv[1]
model_path = sys.argv[2]
result_path = sys.argv[3]

INPUT_SHAPE = (48, 48, 1)
EXPRESSIONS_INDEX = {
        Exp.ANGER: 0,
        Exp.FEAR: 1,
        Exp.HAPPINESS: 2,
        Exp.NEUTRAL: 3,
        Exp.SADNESS: 4,
        Exp.SURPRISE: 5
    }
STANDARD_COUNT = 20

# - Datasets ------------------------------------------------------------------
# -----------------------------------------------------------------------------

print(' LOADING CSV DATAS                                            ')
DATAS = FER2013Dataset()
DATAS.load_csv(fer2013_path)
print(' -----                                                   DONE ')

DATAS.shuffle()

# Get some standard pics to compare with.
# standard_pics = {
#         Exp.ANGER: DATAS.get_n_pictures([Exp.ANGER], count=STANDARD_COUNT),
#         Exp.FEAR: DATAS.get_n_pictures([Exp.FEAR], count=STANDARD_COUNT),
#         Exp.HAPPINESS: DATAS.get_n_pictures([Exp.HAPPINESS], count=STANDARD_COUNT),
#         Exp.NEUTRAL: DATAS.get_n_pictures([Exp.NEUTRAL], count=STANDARD_COUNT),
#         Exp.SADNESS: DATAS.get_n_pictures([Exp.SADNESS], count=STANDARD_COUNT),
#         Exp.SURPRISE: DATAS.get_n_pictures([Exp.SURPRISE], count=STANDARD_COUNT),
#     }

standard_pics = DATAS.get_n_pictures([Exp.ANGER], count=STANDARD_COUNT)

# training_pics = DATAS.get_n_pictures([Exp.ANGER, Exp.FEAR, Exp.HAPPINESS,
#                                       Exp.NEUTRAL, Exp.SADNESS, Exp.SURPRISE],
#                                      count=50, start=10)

# training_pics = DATAS.get_n_pictures([Exp.HAPPINESS],
#                                      count=200, start=0)

validation_pics = DATAS.get_n_pictures([Exp.HAPPINESS, Exp.FEAR, Exp.ANGER,
                                       Exp.NEUTRAL, Exp.SADNESS, Exp.SURPRISE],
                                       target="validation",
                                       count=20, start=10)

# - Printing Images List ------------------------------------------------------
# -----------------------------------------------------------------------------

img_csv = open(result_path + 'img.csv', 'w')
for i in range(0, len(standard_pics)):
    img, exp = standard_pics[i]
    height, width, dim = img.shape
    img = img.reshape(height*width*dim)
    img = img.tolist()
    s = 's' + str(i) + ','
    s += ' '.join(str(e) for e in img) + ','
    s += exp.to_str() + '\n'
    img_csv.write(s)
for i in range(0, len(validation_pics)):
    img, exp = validation_pics[i]
    height, width, dim = img.shape
    img = img.reshape(height*width*dim)
    img = img.tolist()
    s = 'v' + str(i) + ','
    s += ' '.join(str(e) for e in img) + ','
    s += exp.to_str() + '\n'
    img_csv.write(s)

# - Models --------------------------------------------------------------------
# -----------------------------------------------------------------------------

m_ha = load_model(model_path)

# - Get Prediction ------------------------------------------------------------
# -----------------------------------------------------------------------------

PREDICT_TUPLES = []

# Init predict tuples with first expression class
# model, exp = models[0]
model, exp = (m_ha, Exp.HAPPINESS)
X1 = []
X2 = []
print("Predict for : "+exp.to_str())

# Prediction datasets construction
for j in range(0, len(validation_pics)):
    sum = 0
    for k in range(0, STANDARD_COUNT):
        img, exp = standard_pics[k]
        X1.append(img)
    for k in range(0, STANDARD_COUNT):
        img, exp = validation_pics[j]
        X2.append(img)

X1 = numpy.asarray(X1)
X2 = numpy.asarray(X2)
sums = model.predict([X1, X2], verbose=1)

print(sums)

sums = sums.reshape(STANDARD_COUNT*len(validation_pics))
sums = sums.tolist()

i = 0
while i < len(sums):
    img, exp = validation_pics[int(i/STANDARD_COUNT)]
    s = exp.to_str()
    for j in range(0, STANDARD_COUNT):
        exp1 = "D"
        if sums[i] >= 0.5:
            exp1 = "S"
        s += " " + exp1
        i += 1
    print(s)

predict_csv = open(result_path + 'predict.csv', 'w')
pos = 0
for j in range(0, len(validation_pics)):
    for k in range(0, STANDARD_COUNT):
        s = 's'+str(k)+','+'v'+str(j)+','+str(sums[pos])+'\n'
        predict_csv.write(s)
        pos += 1
