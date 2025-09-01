import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

print("All packages imported successfully")

# === Step 1: Generate driving data ===
def generate_driving_data(seq_length=10, num_samples=1000):
    X, y = [], []
    for _ in range(num_samples):
        seq = []
        for t in range(seq_length):
            speed = np.random.uniform(0, 120)
            throttle = np.random.uniform(0, 1)
            steering = np.random.uniform(-30, 30)
            brake = np.random.uniform(0, 1)
            seq.append([speed, throttle, steering, brake])
        X.append(seq)

        # Simple rule-based labeling
        if brake > 0.7:
            decision = 0  # Brake
        elif throttle > 0.7 and speed < 100:
            decision = 1  # Accelerate
        elif steering > 15:
            decision = 2  # Turn Right
        elif steering < -15:
            decision = 3  # Turn Left
        else:
            decision = 4  # Maintain
        y.append(decision)

    return np.array(X), np.array(y)

# === Step 2: Prepare data ===
seq_length, features, num_classes = 10, 4, 5
X, y = generate_driving_data(seq_length)
y = tf.keras.utils.to_categorical(y, num_classes)

# === Step 3: Build LSTM model ===
model = Sequential([
    LSTM(64, input_shape=(seq_length, features)),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training LSTM model...")
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
print("Training complete.")

# === Step 4: Test with sample input ===
sample_input = np.array([
    [60, 0.6,  2, 0.1],
    [62, 0.7,  3, 0.0],
    [65, 0.8,  4, 0.0],
    [67, 0.9,  5, 0.0],
    [70, 0.8,  6, 0.0],
    [72, 0.7,  7, 0.0],
    [75, 0.6,  8, 0.0],
    [78, 0.5,  9, 0.0],
    [80, 0.4, 10, 0.0],
    [82, 0.3, 12, 0.0]
]).reshape((1, seq_length, features))

actions = ["Brake", "Accelerate", "Turn Right", "Turn Left", "Maintain"]

prediction = model.predict(sample_input)
predicted_action = actions[np.argmax(prediction)]

print("\n=== Sample Prediction ===")
print("Raw Prediction Probabilities:", prediction)
print("Predicted Driving Action:", predicted_action)
