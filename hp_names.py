from enum import Enum, auto


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class HP(AutoName):
    UNITS_PER_LAYER = auto()
    N_HID_LAYERS = auto()
    ACTIV_FUN = auto()
    OPTIMIZER = auto()
    LR = auto()
    MOMENTUM = auto()
    LAMBDA_L2 = auto()
    MB = auto()
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
    loss_vl_last = auto()
    mse_vl_last = auto()
    loss_tr_last = auto()
    mse_tr_last = auto()
    accuracy = auto()
    epochs_done = auto()