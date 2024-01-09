# Imports
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from keras_core.models import Model
from keras_core.layers import Dense, Conv2D, ReLU, BatchNormalization, Add, Input, Flatten, AveragePooling2D, Resizing, MaxPooling2D, Dropout
from keras_core.optimizers import Adam, SGD, Adamax
from keras_core.optimizers.schedules import ExponentialDecay

# Image reading
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_val = pd.read_csv('./data/val.csv')

df_train['Data'] = df_train['Image'].apply(lambda x: imread(f'./data/train_images/{x}'))
df_test['Data'] = df_test['Image'].apply(lambda x: imread(f'./data/test_images/{x}'))
df_val['Data'] = df_val['Image'].apply(lambda x: imread(f'./data/val_images/{x}'))

# Data standardization
scaler = StandardScaler()
df_train['DataNorm'] = df_train['Data'].apply(lambda x: scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape))
df_val['DataNorm'] = df_val['Data'].apply(lambda x: scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape))
df_test['DataNorm'] = df_test['Data'].apply(lambda x: scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape))

# Data reshaping
X_train = np.stack(df_train['DataNorm']).reshape(-1, 64, 64, 3)
X_val = np.stack(df_val['DataNorm']).reshape(-1, 64, 64, 3)

y_train = df_train['Class'].to_numpy()
y_val = df_val['Class'].to_numpy()


# Resnet block
def resnet_block(x, filters, strides_number=1):
    y = Conv2D(filters, kernel_size=(3, 3), strides=strides_number)(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Conv2D(filters, kernel_size=(3, 3), strides=1)(y)
    y = BatchNormalization()(y)

    x = Conv2D(filters, kernel_size=(1, 1), strides=strides_number)(x) if strides_number > 1 else x
    
    out = Add()([x, y])
    return ReLU()(out)


# Resnet architecture
def resnet(num_classes):
    inputs = Input(shape=(64, 64, 3))
    x = Resizing(224, 224, interpolation='bilinear', name='resize')(inputs)

    x = Conv2D(64, kernel_size=(3, 3), strides=1)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=128, strides_number=2)
    x = resnet_block(x, filters=128)
    x = resnet_block(x, filters=256, strides_number=2)
    x = resnet_block(x, filters=256)

    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


# Resnet training
clf = resnet(100)
clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
clf.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), verbose=1)

# Save resnet model
clf.save('./resnet_model.keras')


# Custom conv model block
def conv_block(x, filters):
    y = Conv2D(filters, kernel_size=(3, 3), activation='relu')(x)
    y = MaxPooling2D()(y)
    return BatchNormalization()(y)


# Custom convolutional network
def conv_net(num_classes):
    inputs = Input(shape=(64, 64, 3))

    x = conv_block(inputs, filters=64)
    x = Dropout(0.25)(x)
    x = conv_block(x, filters=64)
    x = Dropout(0.35)(x)
    x = conv_block(x, filters=128)
    x = Dropout(0.45)(x)
    x = conv_block(x, filters=256)
    x = Dropout(0.45)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


# Learning rate scheduler
lr_scheduler = ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
optimizer = Adam(lr_scheduler)

# Train conv network
clf = conv_net(100)
clf.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
clf.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1)

# Write submission
X_test = np.stack(df_test['DataNorm']).reshape(-1, 64, 64, 3)
df_test['Class'] = clf.predict_classes(X_test).argmax(axis=-1)

df_test[['Image', 'Class']].to_csv('./prediction.csv', index=False)

# Hyperparameter tuning
ht_optimizers = [Adam, SGD, Adamax]
ht_learning_rates = [0.001, 0.0005, 0.0001]
ht_epochs = [50, 75, 100]
ht_results = []

for ht_optimizer in ht_optimizers:
    for ht_learning_rate in ht_learning_rates:
        for ht_epoch in ht_epochs:
            print(f'Now training on {ht_optimizer_instance.name} optimizer with base learning rate {ht_learning_rate} for {ht_epoch} epochs.')
            ht_clf = conv_net(100)

            lr_scheduler = ExponentialDecay(initial_learning_rate=ht_learning_rate, decay_steps=10000, decay_rate=0.9)
            ht_optimizer_instance = ht_optimizer(lr_scheduler)
            
            ht_clf.compile(optimizer=ht_optimizer_instance, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            ht_clf.fit(X_train, y_train, epochs=ht_epoch, validation_data=(X_val, y_val), verbose=1)

            ht_results.append(ht_clf.evaluate(X_val, y_val)[1])
            del ht_clf


# Make predictions
predictions = clf.predict_classes(X_test).argmax(axis=-1)

# Confusion matrix
conf_matrix = confusion_matrix(df_val['Class'], predictions)

plt.figure(figsize=(14, 14))
plt.imshow(conf_matrix)
plt.xticks(np.arange(len(conf_matrix)), fontsize=7, rotation='vertical')
plt.yticks(np.arange(len(conf_matrix)), fontsize=7)
plt.show()

