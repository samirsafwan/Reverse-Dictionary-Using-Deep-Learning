import tensorflow as tf
import params

def attention(inputs, attention_size):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
   
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    this must be a tensor of shape: `[max_time, batch_size, cell.output_size]`.
        attention_size: Linear size of the Attention weights.
        Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
    """

    inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = params.HIDDEN_LAYERS

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)
    alphas = tf.nn.softmax(vu)

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    return output, alphas