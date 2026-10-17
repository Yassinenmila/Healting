import tensorflow as tf

print("GPU détecté :", tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal((2000, 2000))
    b = tf.random.normal((2000, 2000))
    c = tf.matmul(a, b)

print("Appareil utilisé :", c.device)
print("Calcul terminé :", c.shape)