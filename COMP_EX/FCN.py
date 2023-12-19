from main import *
from comp.MODEL_COMP_SEMI import *
# from MODEL_COMP_UN import *

# UN
# epoch1, epoch2, epoch3 = 94, 40, 85    # per:7(completed)
# epoch1, epoch2, epoch3 = 96, 42, 30  # per:3(completed)
# epoch1, epoch2, epoch3 = 86, 28, 21   # per:1(completed)
# epoch1, epoch2, epoch3 =97,79,65   # per:5(completed)
# epoch1, epoch2, epoch3 = 100,81,100  # per:9(completed)


# SEMI
# epoch1, epoch2, epoch3 = 3,5,7   # per:7(completed)
# epoch1, epoch2, epoch3 =9,2,6    # per:3(completed)
# epoch1, epoch2, epoch3 = 10, 4, 2   # per:1(completed)
epoch1, epoch2, epoch3 = 8,8,6   # per:5(completed)
# epoch1, epoch2, epoch3 = 9, 8, 10   # per:9(completed)

# epoch1, epoch2, epoch3 = 6,3,2    # bo20(completed)
# epoch1, epoch2, epoch3 = 10,9,10    # bo5(completed)
# epoch1, epoch2, epoch3 = 8,5,4    # bo19(completed)
# epoch1, epoch2, epoch3 = 7,6,10    # bx24(completed)



set_class_1, _, weight_per_1, _ = BUILD_WEIGHT_MAT(y_train_1, rho=rho, additional_weights=[], show_report=False)
set_class_2, _, weight_per_2, _ = BUILD_WEIGHT_MAT(y_train_2, rho=rho, additional_weights=[], show_report=False)
set_class_3, _, weight_per_3, _ = BUILD_WEIGHT_MAT(y_train_3, rho=rho, additional_weights=[], show_report=False)

#FCN
input_dim = para_size*num_features*channels

tf.set_random_seed(1)
np.random.seed(1)
OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
model1 = Sequential()
model1.add(Dense(input_dim, input_dim=para_size*num_features*channels))
model1.add(LeakyReLU())
model1.add(Dense(64))
model1.add(LeakyReLU())
model1.add(Dense(128))
model1.add(LeakyReLU())
model1.add(Dropout(dropout))
model1.add(Dense(num_classes,
                 activation='softmax'))
model1.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])
training_1 = model1.fit(x_train_1, to_categorical(y_train_1, num_classes), epochs=epoch1, batch_size=batch_size, verbose=1,
           validation_data=(x_val_1, to_categorical(y_val_1, num_classes)),
           class_weight=
           {set_class_1[0]: 1 * weight_per_1[0],
            set_class_1[1]: 1 * weight_per_1[1],
            set_class_1[2]: 1 * weight_per_1[2]})

y_pred_1 = np.argmax(model1.predict(x_val_1), axis=1)
f1_ave_1 = f1_score(y_val_1, y_pred_1, average='macro')
print(f1_ave_1)

accuracies_2 = training_1.history['val_acc']
losses_2 = training_1.history['val_loss']
loc1 = np.where(accuracies_2 == np.max(accuracies_2))[0]
print(loc1[np.argmin(np.array(losses_2)[loc1])] + 1)
W1 = model1.get_weights()


tf.set_random_seed(1)
np.random.seed(1)
OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
model2 = Sequential()
model2.add(Dense(input_dim, input_dim=para_size*num_features*channels))
model2.add(LeakyReLU())
model2.add(Dense(64))
model2.add(LeakyReLU())
model2.add(Dense(128))
model2.add(LeakyReLU())
model2.add(Dropout(dropout))
model2.add(Dense(num_classes, activation='softmax'))
model2.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])
training_2 = model2.fit(x_train_2, to_categorical(y_train_2, num_classes), epochs=epoch2, batch_size=batch_size, verbose=1,
           validation_data=(x_val_2, to_categorical(y_val_2, num_classes)),
           class_weight=
           {set_class_2[0]: 1 * weight_per_2[0],
            set_class_2[1]: 1 * weight_per_2[1],
            set_class_2[2]: 1 * weight_per_2[2]})

y_pred_2 = np.argmax(model2.predict(x_val_2), axis=1)
f1_ave_2 = f1_score(y_val_2, y_pred_2, average='macro')
print(f1_ave_2)

accuracies_2 = training_2.history['val_acc']
losses_2 = training_2.history['val_loss']
loc1 = np.where(accuracies_2 == np.max(accuracies_2))[0]
print(loc1[np.argmin(np.array(losses_2)[loc1])] + 1)
W2 = model2.get_weights()


tf.set_random_seed(1)
np.random.seed(1)
OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
model3 = Sequential()
model3.add(Dense(input_dim, input_dim=para_size*num_features*channels))
model3.add(LeakyReLU())
model3.add(Dense(64))
model3.add(LeakyReLU())
model3.add(Dense(128))
model3.add(LeakyReLU())
model3.add(Dropout(dropout))
model3.add(Dense(num_classes, activation='softmax'))
model3.compile(optimizer=OPT, loss='categorical_crossentropy', metrics=['accuracy'])
training_3 = model3.fit(x_train_3, to_categorical(y_train_3, num_classes), epochs=epoch3, batch_size=batch_size, verbose=1,
           validation_data=(x_val_3, to_categorical(y_val_3, num_classes)),
           class_weight=
           {set_class_3[0]: 1 * weight_per_3[0],
            set_class_3[1]: 1 * weight_per_3[1],
            set_class_3[2]: 1 * weight_per_3[2]})

y_pred_3 = np.argmax(model3.predict(x_val_3), axis=1)
f1_ave_3 = f1_score(y_val_3, y_pred_3, average='macro')
print(f1_ave_3)

accuracies_2 = training_3.history['val_acc']
losses_2 = training_3.history['val_loss']
loc1 = np.where(accuracies_2 == np.max(accuracies_2))[0]
print(loc1[np.argmin(np.array(losses_2)[loc1])] + 1)
W3 = model1.get_weights()


tf.set_random_seed(1)
np.random.seed(1)

len_param = len(W1)
new_param = []
print(np.shape(W1[1]), type(W1[1]))

for i in range(len_param):
    new_pa = (W1[i] * (f1_ave_1 ** 1) + W2[i] * (f1_ave_2 ** 1) + W3[i] * (f1_ave_3 ** 1))\
            / (1 * (f1_ave_1 ** 1) + 1 * (f1_ave_2 ** 1) + 1 * (f1_ave_3 ** 1))
    #new_pa = (dis_wb1[i] * (fscore1 ** 1) + dis_wb2[i] * (fscore2 ** 1))\
    #        / (1 * (fscore1 ** 1) + 1 * (fscore2 ** 1))
    # new_pa = np.mean(np.array([dis_wb1[i], dis_wb2[i], dis_wb3[i]]), axis=0)
    new_param.append(new_pa)

model_all = Sequential()
model_all.add(Dense(input_dim, input_dim=para_size*num_features*channels))
model_all.add(LeakyReLU())
model_all.add(Dense(64))
model_all.add(LeakyReLU())
model_all.add(Dense(128))
model_all.add(LeakyReLU())
model_all.add(Dropout(dropout))
model_all.add(Dense(num_classes, activation='softmax'))

model_all.set_weights(new_param)

x4, y4 = x_test * 1, y_test * 1
y4_pred = np.argmax(model_all.predict(x4), axis=1)
print(classification_report(y4, y4_pred))
f1_all = f1_score(y4, y4_pred, average=None)
f1_ave_0 = np.sum(f1_all) / num_classes
print(f1_all)
print(f1_ave_0)

yy = list(range(6307, 6589))
yyy = dataset.Y_[yy].reshape(len(yy), -1)
AAA = dataset.A[yy].reshape(len(yy), -1)
y_fcn = np.argmax(model_all.predict(AAA), axis=1)
np.set_printoptions(threshold=np.inf)
print(y_fcn)