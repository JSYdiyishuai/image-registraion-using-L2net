from keras import backend as K
from keras.layers.core import Layer


class LRN(Layer):

    def __init__(self, alpha=256,k=0,beta=0.5,n=256, **kwargs):
        super(LRN, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def call(self, x, mask=None):

            s = K.shape(x)
            b = s[0]
            r = s[1]
            c = s[2]
            ch = s[3]

            half_n = self.n // 2

            input_sqr = K.square(x)

            extra_channels = K.zeros((b, r, c, ch + 2 * half_n))
            input_sqr = K.concatenate([extra_channels[:, :, :, :half_n],input_sqr, extra_channels[:, :, :, half_n + ch:]], axis = 3)

            scale = self.k
            norm_alpha = self.alpha / self.n
            for i in range(self.n):
                scale += norm_alpha * input_sqr[:, :, :, i:i+ch]
            scale = scale ** self.beta
            x = x / scale
            
            return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

