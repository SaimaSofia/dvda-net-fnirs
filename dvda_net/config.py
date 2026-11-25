"""Training and experiment settings."""

import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# UFFT three-class labels remapped to 0/1/2
ALLOWED_LABELS = (1, 2, 3)
LABEL_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["Class 1", "Class 2", "Class 3"]

BATCH_SIZE = 64
PRETRAIN_EPOCHS = 40
FINETUNE_EPOCHS = 20
TGT_TRAIN_RATIO = 0.8

LAMBDA_CONTRAST = 0.02
LAMBDA_ALIGN = 0.02
LAMBDA_STYLE = 0.005
USE_DUAL_VIEW = True

# 6 held-out subjects per fold -> 5 folds on 30 subjects (24 source / 6 target)
TEST_SUBJECTS_PER_FOLD = 6
MAX_FOLDS = None
