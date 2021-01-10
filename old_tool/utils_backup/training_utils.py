import numpy as np
from random import shuffle
from old_tool.utils_backup.analyzing_data import plot_training_result, log_confusion_matrix, cm_4_exp
from old_tool.utils_backup.generic_utils import *
import tensorflow as tf


def _data_struct(data, split):
    data_integer = [np.where(r == 1)[0][0] for r in data]
    struct_data = {}
    for i in range(len(data_integer)):
        if data_integer[i] not in struct_data:
            struct_data[data_integer[i]] = {"index": [i], "base": 0, "offset": None}
        else:
            struct_data[data_integer[i]]["index"].append(i)

    for el in struct_data:
        if struct_data[el]["offset"] is None:
            struct_data[el]["offset"] = int(len(struct_data[el]["index"]) * split)

    return struct_data


def balanced_split_train_test(Y_data, split):
    struct_data = _data_struct(Y_data, split)
    train_ind = []
    test_ind = []
    for el in struct_data:
        temp_base = struct_data[el]["base"]
        temp_base_plus_offset = struct_data[el]["base"] + struct_data[el]["offset"]
        train_ind.extend(struct_data[el]["index"][:temp_base])
        test_ind.extend(struct_data[el]["index"][temp_base:temp_base_plus_offset])
        train_ind.extend(struct_data[el]["index"][temp_base_plus_offset:])
        struct_data[el]["base"] = temp_base_plus_offset

    shuffle(train_ind)
    shuffle(test_ind)

    del struct_data

    return train_ind, test_ind


def balanced_k_fold(Y_data, folder):
    split = 1 / folder
    struct_data = _data_struct(Y_data, split)
    for f in range(folder):
        train_ind = []
        test_ind = []
        for el in struct_data:
            temp_base = struct_data[el]["base"]
            temp_base_plus_offset = struct_data[el]["base"] + struct_data[el]["offset"]
            train_ind.extend(struct_data[el]["index"][:temp_base])
            test_ind.extend(struct_data[el]["index"][temp_base:temp_base_plus_offset])
            train_ind.extend(struct_data[el]["index"][temp_base_plus_offset:])
            struct_data[el]["base"] = temp_base_plus_offset

        shuffle(train_ind)
        shuffle(test_ind)
        yield train_ind, test_ind

    del struct_data


def training(arguments, model_class, total_train_data, total_train_labels, validation=None, save_model=False,
             plot=False, tf_callback=None):

    # Instantiate the class
    model = model_class.build()

    # Split the training dataset into training and validation set
    if validation is not None:
        ind_train, ind_val = validation
        X_train, X_val = total_train_data[ind_train], total_train_data[ind_val]
        y_train, y_val = total_train_labels[ind_train], total_train_labels[ind_val]

        history = model.fit(x=X_train, y=y_train, batch_size=arguments.batch_size, epochs=arguments.epochs,
                            validation_data=(X_val, y_val), callbacks=[tf_callback])

        del X_train, X_val, y_train, y_val

        result = {"train_acc": history.history['acc'], "train_loss": history.history['loss'],
                  "val_acc": history.history['val_acc'], "val_loss": history.history['val_loss']}

    else:

        history = model.fit(x=total_train_data, y=total_train_labels, batch_size=arguments.batch_size,
                            epochs=arguments.epochs, callbacks=[tf_callback])

        result = {"train_acc": history.history['acc'], "train_loss": history.history['loss']}

    if save_model:
        # save model and architecture to single file
        model.save(main_path + 'model_saved/{}_m{}_e{}_b{}.h5'.format(arguments.output_model, arguments.model,
                                                                      arguments.epochs, arguments.batch_size))
        print_log("Saved 'model_saved/{}_m{}_e{}_b{}.h5' to disk".format(arguments.output_model, arguments.model,
                                                                         arguments.epochs, arguments.batch_size),
                  print_on_screen=True)

    if plot:
        plot_training_result(history.history)

    # Return result and the trained model
    return result, model


def test(model, X_test, y_test, fw_cm=(None, None)):

    if fw_cm[0] is not None:
        # Use the model to predict the values from the test dataset
        test_pred_raw = model.predict(X_test)
        test_pred = np.argmax(test_pred_raw, axis=1)
        y_test_int = np.argmax(y_test, axis=1)
        log_confusion_matrix(test_pred, y_test_int, fw_cm[0], fw_cm[1])

    # The TEST part
    result = model.evaluate(x=X_test, y=y_test)
    print('TEST RESULT: {} = {:.4f} / {} = {:.4f}'.format(model.metrics_names[0], result[0],
                                                          model.metrics_names[1], result[1]))

    return {"test_acc": result[1], "test_loss": result[0]}


def training_EXP(model_class, train_ds, epochs, val_ds=None, tf_callback=None):

    # Build and Initialize the model
    model_class.build()
    model = model_class.model

    # Split the training dataset into training and validation set
    if val_ds is not None:

        history = model.fit(x=train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tf_callback])

        result = {"train_acc": history.history['acc'], "train_loss": history.history['loss'],
                  "val_acc": history.history['val_acc'], "val_loss": history.history['val_loss']}

    else:

        history = model.fit(x=train_ds, epochs=epochs, callbacks=[tf_callback])

        result = {"train_acc": history.history['acc'], "train_loss": history.history['loss']}

    # Return result and the trained model
    return result, model


def test_EXP(model_class, test_ds, fw_cm=(None, None)):

    if not model_class.trained or model_class.model is None:
        print("Model needs to be trained before testing, exiting...")
        exit()

    # Gen Confusion Matrix
    if fw_cm[0] is not None:
        cm_4_exp(model_class, test_ds, fw_cm)

    # The TEST part
    model = model_class.model
    result = model.evaluate(x=test_ds)
    print('TEST RESULT: {} = {:.4f} / {} = {:.4f}'.format(model.metrics_names[0], result[0],
                                                          model.metrics_names[1], result[1]))

    return {"test_acc": result[1], "test_loss": result[0]}

