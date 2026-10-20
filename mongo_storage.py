from pymongo import MongoClient

# Replace this with your MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)

db = client["solar_energy_db"]

cleaned_collection = db["cleaned_solar_data"]
model_collection = db["trained_models"]

# Test the connection
client.admin.command("ping")

print("Connected to MongoDB successfully!")