from hp_names import CFG

CUP_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 2,
    CFG.INPUT_DIM: 10,
    CFG.TASK_TYPE: 'regression',
    CFG.MODEL_TYPE: 'CUP_custom_nn',
    CFG.DATASET_FILENAME: 'dev_split.csv',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\ml-cup21-internal_splits\\',
    CFG.CV_NUM_SPLITS: 3,
}
MONK1_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification',
    CFG.MODEL_TYPE: 'monk_custom_nn',
    CFG.DATASET_FILENAME: 'monks-1.train',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\monk\\',
}
MONK2_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification',
    CFG.MODEL_TYPE: 'monk_custom_nn',
    CFG.DATASET_FILENAME: 'monks-2.train',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\monk\\',
}
MONK3_CUSTOM_NET_CFG = {
    CFG.OUT_DIM: 1,
    CFG.INPUT_DIM: 17,
    CFG.TASK_TYPE: 'classification',
    CFG.MODEL_TYPE: 'monk_custom_nn',
    CFG.DATASET_FILENAME: 'monks-3.train',
    CFG.DATASET_DIR: '.\\datasplitting\\assets\\monk\\',
}