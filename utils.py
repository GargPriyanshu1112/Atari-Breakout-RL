def copy_weights(m1, m2):
    m1.set_weights(m2.get_weights())
    return m1
