import math

import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflowjs as tfjs
import random

num_categories = 18


def pretentious_phi(score):
    return 1 / (1 + math.exp(-2 * (score - 5)))


root_number = 2
norm_limit = math.pow(num_categories, 1 / 3 * root_number)


def normalize_tgcs_vec(tgcs_vec):
    return (math.pow(pretentious_phi(tgcs_vec[0]), 1 / root_number),
            math.pow(pretentious_phi(tgcs_vec[1]), 1 / root_number),
            math.pow(pretentious_phi(tgcs_vec[2]), 1 / root_number),
            math.pow(pretentious_phi(tgcs_vec[3]), 1 / root_number),
            math.pow(tgcs_vec[4], 1 / 3 * root_number) / norm_limit)


sample_size = 100_000

X = []
for i in range(sample_size):
    score_mgmt = random.uniform(0, 10)
    score_benefit = random.uniform(0, 10)
    score_difficulty = random.uniform(0, 10)
    score_time = random.uniform(0, 10)
    norm = random.uniform(0, norm_limit)
    as_tuple = normalize_tgcs_vec((score_mgmt, score_benefit, score_difficulty, score_time, norm))
    X.append(as_tuple)

score_functions = [
    lambda tgcs_vec: (tgcs_vec[0] + tgcs_vec[1] + tgcs_vec[2] + tgcs_vec[3]) / 4 * tgcs_vec[4],
    lambda tgcs_vec: (min(tgcs_vec[0], tgcs_vec[1], tgcs_vec[2]) * tgcs_vec[4]),
    lambda tgcs_vec: (max(tgcs_vec[0], tgcs_vec[1], tgcs_vec[2]) * tgcs_vec[4]),
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[1] * tgcs_vec[2] * tgcs_vec[3], 1 / 4) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[1] * tgcs_vec[2] * tgcs_vec[3], 1 / 3) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[2] * tgcs_vec[3], 1 / 3) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[1] * tgcs_vec[3], 1 / 3) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[1] * tgcs_vec[2], 1 / 3) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[2] * tgcs_vec[3], 1 / 2) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[1] * tgcs_vec[3], 1 / 2) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[1] * tgcs_vec[2], 1 / 2) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[3], 1 / 2) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[2], 1 / 2) * tgcs_vec[4],
    lambda tgcs_vec: math.pow(tgcs_vec[0] * tgcs_vec[1], 1 / 2) * tgcs_vec[4],
    lambda tgcs_vec: tgcs_vec[3] * tgcs_vec[4],
    lambda tgcs_vec: tgcs_vec[2] * tgcs_vec[4],
    lambda tgcs_vec: tgcs_vec[1] * tgcs_vec[4],
    lambda tgcs_vec: tgcs_vec[0] * tgcs_vec[4],
    lambda tgcs_vec: tgcs_vec[4]
]


def manual_score(x):
    net_score = 0
    for score_function in score_functions:
        net_score += score_function(x)

    net_score /= len(score_functions)
    return net_score


print("Generating y")
y = [manual_score(x) for x in X]

print("Train-test split")
train_size = math.floor(len(X) * 0.8)
X_train = tf.constant(X[:train_size])
y_train = tf.constant(y[:train_size])
X_test = tf.constant(X[train_size:])
y_test = tf.constant(y[train_size:])

model = tf.keras.Sequential([
    Dense(2),
    Dense(3, activation='sigmoid'),
    Dense(1)
])

print("Compiling model")
model.compile(
    optimizer='sgd',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

print("Fitting model")
model.fit(
    X_train,
    y_train,
    batch_size=50,
    epochs=25
)

print(model.evaluate(X_test, y_test, batch_size=50))
tfjs.converters.save_keras_model(model, 'tgcs_model/')
# model.save("tgcs_model/")
#
# new_model = tf.keras.models.load_model('tgcs_model/')
# print(new_model.evaluate(X_test, y_test, batch_size=50))
#
# new_model = tf.keras.models.load_model('tgcs_model/')
# print(new_model.predict(tf.constant([
#     normalize_tgcs_vec((5, 5, 5, 5, 0.5251396430119836)),
#     normalize_tgcs_vec((5, 5, 5, 5, 0.2059767143907118))
# ])))
