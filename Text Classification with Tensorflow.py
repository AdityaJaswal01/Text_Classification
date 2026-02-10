import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
train_data , validation_data, test_data= tfds.load(name="imdb_reviews", split=('train[:60%]','train[:60%]', 'test'), as_supervised=True)
train_data()
trian_example_batch , train_label_batch = next(iter(train_data.batch(10)))
trian_example_batch()
train_label_batch()
embedding ="https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding,input_shape = [] , dtype=tf.string ,trainable=True)
hub_layer(trian_example_batch[:3])
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16 , activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
history=model.fit(train_data.shuffle(10000).batch(100),epochs=25,validation_data=validation_data.batch(100),verbose=1)
results = model.evaluate(test_data.batch(100),verbose=1)
for name , values in zip(model.metrics_names, results):
    print("%s: %s" % (name, values))