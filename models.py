import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class add_cls(layers.Layer):
    def __init__(self, dims):
        super(add_cls, self).__init__()
        self.dims = dims
        self.cls = self.add_weight('cls', (1, 1, dims))
        
    def build(self, inputs_shape):
        self.patchs = inputs_shape[1] * inputs_shape[2]
        self.pos_embed = self.add_weight('pos', (self.patchs, self.dims)) * 2e-2
        
    def call(self, inputs):
        b = tf.shape(inputs)[0]
        x = tf.reshape(inputs, (b, -1, self.dims)) + self.pos_embed
        return tf.concat([tf.tile(self.cls, (b, 1, 1)), x], 1)


class Transformer(layers.Layer):
    def __init__(self, dims, heads):
        super(Transformer, self).__init__()
        self.dims = dims
        
        self.LN = layers.LayerNormalization()
        self.qkv = layers.Dense(3 * dims)
        self.mha = layers.MultiHeadAttention(heads, dims)
        
        leakyReLU = layers.LeakyReLU()
        self.FFN = Sequential([
            layers.Dense(2 * dims, activation=leakyReLU),
            layers.Dense(dims, activation=leakyReLU),
        ])
        
    def call(self, inputs):
        x = self.LN(inputs)
        
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, -1)
        
        z = self.mha(q, k, v) + x
        
        outputs = self.FFN(z) + z             
        return outputs
        
    def get_attention_weight(self, inputs):
        x = self.LN(inputs)
        
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, -1)

        z, w = self.mha(
            q, k, v, 
            return_attention_scores=True
        )
        z = z + x
        outputs = self.FFN(z) + z 
        return outputs, w

    def get_Kz(self, inputs):
        x = self.LN(inputs)
        
        qkv = self.qkv(x)
        q, k, v = tf.split(qkv, 3, -1)
        
        z = self.mha(q, k, v) + x
        
        outputs = self.FFN(z) + z             
        return outputs, k


class model_vit(Model):
    def __init__(self, dims, num_classes, heads, num_layers=3):
        super(model_vit, self).__init__(name='vit')
        leakyReLU = layers.LeakyReLU()
        self.encoder = Sequential([
            layers.Conv2D(dims, 2, 2, activation=leakyReLU),
            layers.MaxPool2D(2),
            layers.Conv2D(dims, 2, 2, activation=leakyReLU),
            layers.MaxPool2D(2),
            add_cls(dims)
        ])
        self.mha = [Transformer(dims, heads) for _ in range(num_layers)]
        self.decoder = Sequential([
            layers.Dense(2 * dims, activation=leakyReLU),
            layers.Dense(num_classes)
        ])
    
    def call(self, inputs):
        z = self.encoder(inputs)
        for mha_i in self.mha:
            z = mha_i(z)
        return self.decoder(z[:, 0])
    
    def get_attention_weight(self, inputs):
        z = self.encoder(inputs)
        z_list = [z]
        w_list = []
        for mha_i in self.mha:
            z, w = mha_i.get_attention_weight(z)
            z_list.append(z)
            w_list.append(w)
        return z_list, w_list

    def get_Kz(self, inputs):
        z = self.encoder(inputs)
        Kz_list = []
        for mha_i in self.mha:
            z, Kz = mha_i.get_Kz(z)
            Kz_list.append(Kz)
        return Kz_list


class Creta(layers.Layer):
    def __init__(self, dims, heads, sigma=.1, lambd=.1):
        super(Creta, self).__init__()
        self.dims = dims
        
        self.LN = layers.LayerNormalization()
        self.U = self.add_weight('U', (dims, dims), initializer='orthogonal')
        self.mha = layers.MultiHeadAttention(heads, dims)
        
        leakyReLU = layers.LeakyReLU()
        self.D = self.add_weight('D', (dims, dims), initializer='orthogonal')
        self.sigma = sigma
        self.lambd = lambd
        
    def call(self, inputs):
        x = self.LN(inputs)
        
        z_l = x @ self.U
        z_half = self.mha(z_l, z_l) + x

        z_next = self.sigma * ((z_half @ self.D - z_half) @ tf.transpose(self.D, (1, 0)) - self.lambd)
        return z_next + z_half
    
    def get_attention_weight(self, inputs):
        x = self.LN(inputs)
        
        z_l = x @ self.U
        z_half, w = self.mha(z_l, z_l, return_attention_scores=True) 
        z_half = z_half + x

        z_next = self.sigma * ((z_half @ self.D - z_half) @ tf.transpose(self.D, (1, 0)) - self.lambd)
        return z_next + z_half, w
    
    def get_Uz(self, inputs):
        x = self.LN(inputs)
        
        z_l = x @ self.U
        z_half = self.mha(z_l, z_l) + x

        z_next = self.sigma * ((z_half @ self.D - z_half) @ tf.transpose(self.D, (1, 0)) - self.lambd)
        return z_next + z_half, z_l


class model_crate(Model):
    def __init__(self, dims, num_classes, heads, num_layers=3):
        super(model_crate, self).__init__(name='vit')
        leakyReLU = layers.LeakyReLU()
        self.encoder = Sequential([
            layers.Conv2D(dims, 2, 2, activation=leakyReLU),
            layers.MaxPool2D(2),
            layers.Conv2D(dims, 2, 2, activation=leakyReLU),
            layers.MaxPool2D(2),
            add_cls(dims)
        ])
        self.mha = [Creta(dims, heads) for _ in range(num_layers)]
        self.decoder = Sequential([
            layers.Dense(2 * dims, activation=leakyReLU),
            layers.Dense(num_classes)
        ])
    
    def call(self, inputs):
        z = self.encoder(inputs)
        for mha_i in self.mha:
            z = mha_i(z)
        return self.decoder(z[:, 0])
    
    def get_attention_weight(self, inputs):
        z = self.encoder(inputs)
        z_list = [z]
        w_list = []
        for mha_i in self.mha:
            z, w = mha_i.get_attention_weight(z)
            z_list.append(z)
            w_list.append(w)
        return z_list, w_list

    def get_Uz(self, inputs):
        z = self.encoder(inputs)
        Uz_list = []
        for mha_i in self.mha:
            z, Uz = mha_i.get_Uz(z)
            Uz_list.append(Uz)
        return Uz_list
