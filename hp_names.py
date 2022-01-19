from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class HP(AutoName):
    NUM_UNITS_PER_HID_LAYER = auto()
    NUM_HID_LAYERS = auto()
    ACTIV_FUN = auto()
    OPTIMIZER = auto()
    LEARNING_RATE = auto()
    MOMENTUM = auto()
    LAMBDA_L2 = auto()
    MINI_BATCH_SIZE = auto()
    STOPPING_THRESH = auto()
    EARLY_STOP_ALG = auto()
    PATIENCE = auto()
    MAX_EPOCHS = auto()
    METRIC = auto()  # accuracy, ..
    ERROR_FN = auto()  # MSE, MEE, ..

    def __lt__(self, other):
        return self.name < other.name


class CFG(AutoName):
    OUT_DIM = auto()
    INPUT_DIM = auto()
    TASK_TYPE = auto()


class RES(AutoName):
    last_vl_loss = auto()
    last_vl_mse = auto()
    last_tr_loss = auto()
    last_tr_mse = auto()
    accuracy = auto()
    epochs_done = auto()