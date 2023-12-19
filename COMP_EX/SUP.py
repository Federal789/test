from main import *

rho_0 = rho   # 1:1,3:0.7

# epoch1, epoch2, epoch3 = 28, 40, 52    # per:7(completed)
# epoch1, epoch2, epoch3 = 21, 41, 13  # per:3(completed)
# epoch1, epoch2, epoch3 = 5, 18, 91   # per:1(completed)
# epoch1, epoch2, epoch3 = 24, 61, 35   # per:5(completed)
# epoch1, epoch2, epoch3 = 69, 48, 41   # per:9(completed)


epoch1, epoch2, epoch3 = 1000, 1000, 1000   # per:7
# epoch1, epoch2, epoch3 = 22, 67, 12  # per:3(completed)
# epoch1, epoch2, epoch3 = 41, 11, 23   # per:1(completed)
# epoch1, epoch2, epoch3 = 99, 97, 67   # per:5
# epoch1, epoch2, epoch3 = 93, 100, 54   # per:9



# Fully supervised classifier with the same network architecture as the SSGAN Discriminator
# tf.random.set_seed(1)
tf.compat.v1.set_random_seed(1)
np.random.seed(1)
FS_classifier_1 = build_discriminator_supervised(build_discriminator(sam_shape, dropout))
OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
FS_classifier_1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=OPT)

tr_idx, va_idx = TR[0], VAL[0]

x_train_lab, x_test, x_val, y_train_lab, y_test, y_val = \
    np.array(dataset.A_tv_lab)[tr_idx], np.array(dataset.A_test), np.array(dataset.A_tv_lab)[va_idx], \
    np.array(dataset.Y_tv_lab)[tr_idx], np.array(dataset.Y_test), np.array(dataset.Y_tv_lab)[va_idx]

cs, _, weight_p, num_p = BUILD_WEIGHT_MAT(y_train_lab, rho=rho_0, additional_weights=[], show_report=False)

# One-hot encode labels
y_train_lab = to_categorical(y_train_lab, num_classes=num_classes)
y_val_ = to_categorical(y_val, num_classes=num_classes)

# Train the classifier
training_1 = FS_classifier_1.fit(x=x_train_lab, y=y_train_lab, batch_size=batch_size, epochs=epoch1, verbose=1,
                                 validation_data=(x_val, y_val_),
                                 class_weight={cs[0]: weight_p[0], cs[1]: weight_p[1], cs[2]: weight_p[2]}
                                 )

accuracies_2 = training_1.history['val_acc']
losses_2 = training_1.history['val_loss']
loc1 = np.where(accuracies_2 == np.max(accuracies_2))[0]
print(loc1[np.argmin(np.array(losses_2)[loc1])] + 1)

W1 = FS_classifier_1.get_weights()

'''
losses_1 = training_1.history['val_loss']
accuracies_1 = training_1.history['val_acc']

# Plot classification loss
plt.figure(figsize=(10, 5))
plt.plot(np.array(losses_1), label="Loss")
plt.title("Classification Loss")
plt.legend()
plt.show()

# Plot classification accuracy
plt.figure(figsize=(10, 5))
plt.plot(np.array(accuracies_1), label="Accuracy")
plt.title("Classification Accuracy")
plt.legend()
filePath = './image/per_{}/'.format(per) + str('ACC_1.jpg')
plt.savefig(filePath)
'''

# print('CNN_supervised:')
# x4, y4 = dataset.A_tv_lab, dataset.Y_tv_lab
# y4_pred = FS_classifier_1.predict_classes(x4)
# print(classification_report(y4, y4_pred))

x5, y5 = x_val, y_val
y5_pred = np.argmax(FS_classifier_1.predict(x5), axis=1)
#y111_pred = FS_classifier_1.predict_classes(x5)
print(classification_report(y5, y5_pred))
f1 = f1_score(y5, y5_pred, average=None)
f1_ave_1 = np.sum(f1) / num_classes
print(f1_score(y5, y5_pred, average='macro'))


# tf.random.set_seed(1)
tf.compat.v1.set_random_seed(1)
np.random.seed(1)
FS_classifier_2 = build_discriminator_supervised(build_discriminator(sam_shape, dropout))
OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
FS_classifier_2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=OPT)

tr_idx, va_idx = TR[1], VAL[1]

x_train_lab, x_test, x_val, y_train_lab, y_test, y_val = \
    np.array(dataset.A_tv_lab)[tr_idx], np.array(dataset.A_test), np.array(dataset.A_tv_lab)[va_idx], \
    np.array(dataset.Y_tv_lab)[tr_idx], np.array(dataset.Y_test), np.array(dataset.Y_tv_lab)[va_idx]

cs, _, weight_p, num_p = BUILD_WEIGHT_MAT(y_train_lab, rho=rho_0, additional_weights=[], show_report=False)

# One-hot encode labels
y_train_lab = to_categorical(y_train_lab, num_classes=num_classes)
y_val_ = to_categorical(y_val, num_classes=num_classes)

# Train the classifier
training_2 = FS_classifier_2.fit(x=x_train_lab, y=y_train_lab, batch_size=batch_size, epochs=epoch2, verbose=1,
                                 validation_data=(x_val, y_val_),
                                 class_weight={cs[0]: weight_p[0], cs[1]: weight_p[1], cs[2]: weight_p[2]}
                                 )

accuracies_2 = training_2.history['val_acc']
losses_2 = training_2.history['val_loss']
loc1 = np.where(accuracies_2 == np.max(accuracies_2))[0]
print(loc1[np.argmin(np.array(losses_2)[loc1])] + 1)

W2 = FS_classifier_2.get_weights()

'''
losses_2 = training_2.history['val_loss']
accuracies_2 = training_2.history['val_acc']

# Plot classification loss
plt.figure(figsize=(10, 5))
plt.plot(np.array(losses_2), label="Loss")
plt.title("Classification Loss")
plt.legend()
plt.savefig(r'images/acc_loss/' + 'supervised_loss_2.jpg')


# Plot classification accuracy
plt.figure(figsize=(10, 5))
plt.plot(np.array(accuracies_2), label="Accuracy")
plt.title("Classification Accuracy")
plt.legend()
filePath = './image/per_{}/'.format(per) + str('ACC_2.jpg')
plt.savefig(filePath)
'''

# print('CNN_supervised:')
# x4, y4 = dataset.A_tv_lab, dataset.Y_tv_lab
# y4_pred = FS_classifier_2.predict_classes(x4)
# print(classification_report(y4, y4_pred))

x5, y5 = x_val, y_val
y5_pred = np.argmax(FS_classifier_2.predict(x5), axis=1)
#y111_pred = FS_classifier_1.predict_classes(x5)
print(classification_report(y5, y5_pred))
f1 = f1_score(y5, y5_pred, average=None)
f1_ave_2 = np.sum(f1) / num_classes
print(f1, f1_ave_2)


# tf.random.set_seed(1)
tf.compat.v1.set_random_seed(1)
np.random.seed(1)
FS_classifier_3 = build_discriminator_supervised(build_discriminator(sam_shape, dropout))
OPT = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
FS_classifier_3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=OPT)

tr_idx, va_idx = TR[2], VAL[2]

x_train_lab, x_test, x_val, y_train_lab, y_test, y_val = \
    np.array(dataset.A_tv_lab)[tr_idx], np.array(dataset.A_test), np.array(dataset.A_tv_lab)[va_idx], \
    np.array(dataset.Y_tv_lab)[tr_idx], np.array(dataset.Y_test), np.array(dataset.Y_tv_lab)[va_idx]

cs, _, weight_p, num_p = BUILD_WEIGHT_MAT(y_train_lab, rho=rho_0, additional_weights=[], show_report=False)

# One-hot encode labels
y_train_lab = to_categorical(y_train_lab, num_classes=num_classes)
y_val_ = to_categorical(y_val, num_classes=num_classes)

# Train the classifier
training_3 = FS_classifier_3.fit(x=x_train_lab, y=y_train_lab, batch_size=batch_size, epochs=epoch3, verbose=1,
                                 validation_data=(x_val, y_val_),
                                 class_weight={cs[0]: weight_p[0], cs[1]: weight_p[1], cs[2]: weight_p[2]}
                                 )

accuracies_2 = training_3.history['val_acc']
losses_2 = training_3.history['val_loss']
loc1 = np.where(accuracies_2 == np.max(accuracies_2))[0]
print(loc1[np.argmin(np.array(losses_2)[loc1])] + 1)

W3 = FS_classifier_3.get_weights()
'''
losses_3 = training_3.history['val_loss']
accuracies_3 = training_3.history['val_acc']

# Plot classification loss
plt.figure(figsize=(10, 5))
plt.plot(np.array(losses_3), label="Loss")
plt.title("Classification Loss")
plt.legend()
plt.savefig('images/acc_loss/' + 'supervised_loss_3.jpg')


# Plot classification accuracy
plt.figure(figsize=(10, 5))
plt.plot(np.array(accuracies_3), label="Accuracy")
plt.title("Classification Accuracy")
plt.legend()
filePath = './image/per_{}/'.format(per) + str('ACC_3.jpg')
plt.savefig(filePath)
'''

# print('CNN_supervised:')
# x4, y4 = dataset.A_tv_lab, dataset.Y_tv_lab
# y4_pred = FS_classifier_3.predict_classes(x4)
# print(classification_report(y4, y4_pred))

x5, y5 = x_val, y_val
y5_pred = np.argmax(FS_classifier_3.predict(x5), axis=1)
#y111_pred = FS_classifier_1.predict_classes(x5)
print(classification_report(y5, y5_pred))
f1 = f1_score(y5, y5_pred, average=None)
f1_ave_3 = np.sum(f1) / num_classes
print(f1, f1_ave_3)

tf.compat.v1.set_random_seed(1)
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

NEW_classifier = build_discriminator_supervised(build_discriminator(sam_shape, dropout))
NEW_classifier.set_weights(new_param)


x4, y4 = dataset.test_set()
y4_pred = np.argmax(NEW_classifier.predict(x4), axis=1)
print(classification_report(y4, y4_pred))
f1_all = f1_score(y4, y4_pred, average=None)
f1_ave_0 = np.sum(f1_all) / num_classes
print(f1_all)
print(f1_ave_0)
