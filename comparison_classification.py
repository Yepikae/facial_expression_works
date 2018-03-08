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

if len(sys.argv) < 3:
    print("Gib FER2013 path and created model path pls.")
    exit()

fer2013_path = sys.argv[1]
model_path = sys.argv[2]

BATCH_SIZE = 64
if len(sys.argv) > 3:
    BATCH_SIZE = sys.argv[3]

INPUT_SHAPE = (48, 48, 1)
PIC_PER_ROUND = 50

# - Datasets ------------------------------------------------------------------
# -----------------------------------------------------------------------------

print(' LOADING CSV DATAS                                            ')
DATAS = FER2013Dataset()
DATAS.load_csv(fer2013_path)
print(' -----                                                   DONE ')
DATAS.shuffle()

# - Model ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

expressions = [Exp.HAPPINESS, Exp.ANGER, Exp.SURPRISE,
               Exp.SADNESS, Exp.FEAR, Exp.NEUTRAL]

# Model creation et setting
model = simple_CNN((48, 48, 1))
opt = optimizers.RMSprop(lr=0.001)
model.compile(opt, loss='binary_crossentropy',
              metrics=['accuracy'])


def can_we_proceed(pointer):
    """Return true if we might continue learning. False otherwise."""
    count = 0
    for i, exp in enumerate(expressions):
        if DATAS.get_data_length(exp) > pointer + PIC_PER_ROUND:
            count += 1
    return True if count >= 2 else False

pointer = 0
while can_we_proceed(pointer):
    print(" === Pointer : "+str(pointer)+" ========== ")
    # Get training sets
    training_pics = DATAS.get_n_pictures(expressions,
                                         count=PIC_PER_ROUND,
                                         start=pointer)
    XT1, XT2, YT = CCG.generate_training_set(training_pics)
    # Train the model
    model.fit([XT1, XT2], YT, batch_size=BATCH_SIZE, epochs=1, verbose=1)
    # Increase pointer for next round
    pointer += PIC_PER_ROUND
    # Save model (in case of crash)
    fpath = model_path
    fpath += "tmp_model_"+str(pointer)+".hdf5"
    model.save(fpath)
# Save final model
fpath = model_path
fpath += "final_model.hdf5"
model.save()
