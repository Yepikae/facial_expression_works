"""Hazard Zone."""

from fee.models import simple_CNN
from fee.data import ComparisonClassifGen as CCG
from fee.classification import Expression as Exp
from fee.data import FER2013Dataset
from keras import optimizers
import sys
import tensorflow as tf

# Limit cores usage
from keras import backend as K
config = tf.ConfigProto(intra_op_parallelism_threads=9,
                        inter_op_parallelism_threads=9,
                        allow_soft_placement=True,
                        device_count={'CPU': 9})
session = tf.Session(config=config)
K.set_session(session)

# - Global Vars ---------------------------------------------------------------
# -----------------------------------------------------------------------------

if len(sys.argv) != 3:
    print("Gib FER2013 path and created model path pls.")
    exit()

fer2013_path = sys.argv[1]
model_path = sys.argv[2]

INPUT_SHAPE = (48, 48, 1)
PIC_PER_ROUND = 50
OTHER_PER_ROUND = 10

# - Datasets ------------------------------------------------------------------
# -----------------------------------------------------------------------------

print(' LOADING CSV DATAS                                            ')
DATAS = FER2013Dataset()
DATAS.load_csv(fer2013_path)
print(' -----                                                   DONE ')

DATAS.shuffle()

# - Model -------------------------------------------------------------
# ---------------------------------------------------------------------

expressions = [Exp.HAPPINESS, Exp.ANGER, Exp.SURPRISE,
               Exp.SADNESS, Exp.FEAR, Exp.NEUTRAL]

model = simple_CNN((48, 48, 1))  # A virer
opt = optimizers.RMSprop(lr=0.001)  # A virer
model.compile(opt, loss='binary_crossentropy',  # A virer
              metrics=['accuracy'])  # A virer
# for idexp1, exp1 in enumerate(expressions):
#     exps = [exp1]

max_pic = DATAS.get_data_length(Exp.HAPPINESS)
max_batch = int(max_pic / PIC_PER_ROUND)

for counter in range(0, max_batch):  # A virer
    print(" ROUND : "+str(counter))
    exps = [Exp.HAPPINESS]
    for idexp2, exp2 in enumerate(expressions):
        # if exp1 is not exp2:
        if exp2 is not Exp.HAPPINESS:  # A virer
            exps.append(exp2)
    print(' CREATING MODEL                                           ')
    # model = simple_CNN((48, 48, 1))
    # opt = optimizers.RMSprop(lr=0.001)
    # model.compile(opt, loss='binary_crossentropy',
    #               metrics=['accuracy'])
    print(' -----                                               DONE ')

    # training_pics = DATAS.get_pictures(exps,
    #                                    part_size=0.002,
    #                                    part_index=counter)
    training_pics = DATAS.get_n_pictures([Exp.HAPPINESS],
                                         count=PIC_PER_ROUND,
                                         start=counter)

    training_pics += DATAS.get_n_pictures([Exp.ANGER, Exp.SURPRISE,
                                           Exp.SADNESS, Exp.FEAR, Exp.NEUTRAL],
                                          count=OTHER_PER_ROUND,
                                          start=counter)

    XT1, XT2, YT = CCG.generate_training_set(training_pics, Exp.HAPPINESS)
    model.fit([XT1, XT2], YT, batch_size=64, epochs=1, verbose=1)
fpath = model_path
fpath += "HAPPINESS_TEST_MODEL.hdf5"
model.save(fpath)




# posV = 0
# posT = 0
# for i in range(0, 5):
#     for j in range(0, 10):
#         index = j+i*j
#         print(' CREATING DATASETS                                            ')
#         training_pics, posT = DATAS.get_pictures([Exp.HAPPINESS, Exp.ANGER],
#                                                 part_size=0.02,
#                                                 start=posT)
#         XT1, XT2, YT = CCG.generate_training_set(training_pics, Exp.HAPPINESS)
#         validate_pics, posV = DATAS.get_pictures([Exp.HAPPINESS, Exp.ANGER],
#                                                 target="validation",
#                                                 part_size=0.02,
#                                                 start=posV)
#         XV1, XV2, YV = CCG.generate_training_set(validate_pics, Exp.HAPPINESS)
#         print(' -----                                                   DONE ')
#
#         # TODO :Add right after "YT,""
#         # ----  validation_data=([XV1, XV2], YV),
#
#         model.fit([XT1, XT2], YT, batch_size=64, epochs=1, verbose=1)
#     fpath = "./Results/comparison_classification/models/HAPPINESS_ANGER"+str(i)+".hdf5"
#     model.save(fpath)
#
# validate_pics, posV = DATAS.get_pictures([Exp.SADNESS, Exp.ANGER],
#                                         target="validation", part_size=0.2)
# XV1, XV2, YV = CCG.generate_training_set(validate_pics, Exp.SADNESS)
# model = load_model('/home/remi/Documents/6eSens/Works/Results/comparison_classification/models/HAPPINESS_ANGER4.hdf5')
# scores = model.evaluate([XV1, XV2], YV)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
