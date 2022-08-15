from pyexpat import model
from tabnanny import verbose
from utils import *

class DQN:
    model = tf.keras.Sequential([
        Dense(256, activation='sigmoid',
                              kernel_initializer='random_normal'),
        BatchNormalization(),
        Dense(128, activation='sigmoid', kernel_initializer='random_normal'),
        BatchNormalization(),
        Dense(64, activation='sigmoid', kernel_initializer='random_normal'),
        BatchNormalization(),
        Dense(32,activation='sigmoid', kernel_initializer='random_normal'),
        BatchNormalization(),
        Dense(3,activation='sigmoid', kernel_initializer='random_normal')
    ])
    def __init__(self,lr,input_dim,loss='mse'):
        self.lr = lr
        self.input_dim = input_dim
        self.loss = loss
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)
    def train_on_batch(self,x,y):
        x = np.array(x).reshape((-1, self.input_dim))
        history = self.model.fit(x,y,epochs=10,verbose=False)
        loss = np.sum(history.history['loss'])
        return loss
    def predict(self,sample):
        sample = np.array(sample).reshape((1,self.input_dim))
        return self.model.predict(sample).flatten()
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

model2 = tf.keras.Sequential([
    LSTM(256, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal'),
    BatchNormalization(),
    LSTM(128, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal'),
    BatchNormalization(),
    LSTM(64, dropout=0.1, return_sequences=True, stateful=False, kernel_initializer='random_normal'),
    BatchNormalization(),
    LSTM(32, dropout=0.1, stateful=False, kernel_initializer='random_normal'),
    BatchNormalization(),
    Dense(3, activation='sigmoid')
])

if __name__ == '__main__':
    nn = model2
    nn.build((1,5,13))
    x = np.random.random((1,5,13))
    print(x.shape)
    print(nn.predict(x))
    