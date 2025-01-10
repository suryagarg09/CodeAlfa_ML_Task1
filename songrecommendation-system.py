import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy

# Example user-song dataset (user_id, song_id, repeated_play)
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'song_id': [101, 102, 103, 101, 104, 105, 102, 106, 107],
    'repeated_play': [1, 0, 1, 1, 0, 1, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the format for the Surprise library
reader = Reader(rating_scale=(0, 1))  # 1 for repeated plays, 0 for no repeat
dataset = Dataset.load_from_df(df[['user_id', 'song_id', 'repeated_play']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.25)

# Train the SVD model
svd = SVD()
svd.fit(trainset)

# Make predictions on the test set
predictions = svd.test(testset)

# Calculate accuracy (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"Root Mean Squared Error: {rmse}")

# Example: Recommend a song for a specific user
user_id = 1
song_ids = [101, 102, 103, 104, 105, 106, 107]  # List of all song IDs

# Predict ratings for all songs
recommendations = []
for song_id in song_ids:
    prediction = svd.predict(user_id, song_id)
    recommendations.append((song_id, prediction.est))

# Sort recommendations based on predicted repeated play probability
recommendations.sort(key=lambda x: x[1], reverse=True)

# Output top 3 recommended songs for user_id 1
print(f"Top recommended songs for user {user_id}:")
for song, score in recommendations[:3]:
    print(f"Song ID: {song}, Predicted Score: {score}")
