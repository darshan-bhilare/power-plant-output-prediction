from ucimlrepo import fetch_ucirepo

# Download the dataset from the UCI website
data = fetch_ucirepo(id=294)
X = data.data.features   # the 4 input columns
y = data.data.targets    # the power output column

# Join them into one table and save a copy on your computer
df = X.copy()
df["PE"] = y["PE"]
df.to_csv("power_plant.csv", index=False)

print("Shape (rows, columns):", df.shape)
print()
print(df.head())          # first 5 rows
print()
print(df.describe())      # min, max, average of each column
print()
print("Missing values per column:")
print(df.isnull().sum())  # how many empty cells