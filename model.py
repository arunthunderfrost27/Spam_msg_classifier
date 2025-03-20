import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('./spam_data.csv', encoding='ISO-8859-1')

df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df = df.rename(columns={"v1": "label", "v2": "message"})

X = df['message'].values
Y = df['label'].values
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

max_words = 1000
max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

def RNN():
    inputs = Input(name='inputs', shape=[max_len])
    layer = Embedding(max_words, 50, input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1, name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
model.save('spam_classifier_model.keras')

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

loaded_model = load_model('spam_classifier_model.keras')

def predict_message(message):
    message_seq = tok.texts_to_sequences([message])
    message_pad = sequence.pad_sequences(message_seq, maxlen=max_len, padding='post', truncating='post')
    prediction = loaded_model.predict(message_pad)
    if prediction <= 0.5:
        return "Spam"
    else:
        return "Non-spam"

new_message = input("Enter a message to test: ")
prediction_result = predict_message(new_message)
print(f"The message is: {prediction_result}")
