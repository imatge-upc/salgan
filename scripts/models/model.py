import theano.tensor as T


class Model(object):
    def __init__(self, input_width, input_height, batch_size=32):
        
        self.inputWidth = input_width
        self.inputHeight = input_height

        self.G_lr = None
        self.D_lr = None
        self.momentum = None

        self.net = None
        self.discriminator = None
        self.batch_size = batch_size

        self.D_trainFunction = None
        self.G_trainFunction = None
        self.predictFunction = None
        self.input_var = T.tensor4()
        self.output_var = T.tensor4()
