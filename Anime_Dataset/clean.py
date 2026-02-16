import pandas as pd

# Load the dataset (replace 'input.csv' with your file name)
df = pd.read_csv('Anime_Dataset/rating.csv')

# Remove rows where rating == -1
df_cleaned = df[df["rating"] != -1]

# Save cleaned dataset
df_cleaned.to_csv("Anime_Dataset/cleaned_rating.csv", index=False)

print("Cleaning completed!")
print(df_cleaned.head())
