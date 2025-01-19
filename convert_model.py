import torch

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization, MaxPool2D, Dropout, Permute, Flatten, Dense


# tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
capacity_multiplier = 8

input_path = "model-small.h5"
output_path = "crepe_small.pth"

layers = [1, 2, 3, 4, 5, 6]
filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
widths = [512, 64, 64, 64, 64, 64]
strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

x = Input(shape=(1024,), name='input', dtype='float32')
y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

for l, f, w, s in zip(layers, filters, widths, strides):
    y = Dropout(0.25, name="conv%d-dropout" % l)(MaxPool2D(pool_size=(2, 1), strides=None, padding='valid', name="conv%d-maxpool" % l)(BatchNormalization(name="conv%d-BN" % l)(Conv2D(f, (w, 1), strides=s, padding='same', activation='relu', name="conv%d" % l)(y))))

model = Model(inputs=x, outputs=Dense(360, activation='sigmoid', name="classifier")(Flatten(name="flatten")(Permute((2, 1, 3), name="transpose")(y))))
model.load_weights(input_path)

state_dict = {}

for i in range(1, 7):
    state_dict["conv{}.weight".format(i)] = torch.tensor(model.get_layer("conv{}".format(i)).kernel.numpy()).permute(3, 2, 0, 1)
    state_dict["conv{}.bias".format(i)] = torch.tensor(model.get_layer("conv{}".format(i)).bias.numpy())
    state_dict["conv{}_BN.weight".format(i)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).gamma.numpy())
    state_dict["conv{}_BN.bias".format(i)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).beta.numpy())
    state_dict["conv{}_BN.running_mean".format(i)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).moving_mean.numpy())
    state_dict["conv{}_BN.running_var".format(i)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).moving_variance.numpy())

state_dict["classifier.weight"] = torch.tensor(model.get_layer("classifier").kernel.numpy()).permute(1, 0)
state_dict["classifier.bias"] = torch.tensor(model.get_layer("classifier").bias.numpy())

torch.save(state_dict, output_path)