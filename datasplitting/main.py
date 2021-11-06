from preprocessing import Preprocessing

if __name__ == '__main__':
    # build_train_and_save_model()
    # load_model()
    # lemon_squeeze_cross_validation()
    input_ids, inX, outy = Preprocessing.load_the_dataset()
    k = 5
    folds = Preprocessing.get_random_k_folds(k, input_ids, inX, outy)

