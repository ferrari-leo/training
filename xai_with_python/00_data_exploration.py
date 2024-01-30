# Import packages
import matplotlib.pyplot as plt
from utils import DataLoader

# Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data
print(data.shape)
display(data.head())

# Show general stats
data.info()

# Show histograms
columns = data.columns
for col in columns:
    print(f"col: {col}")
    data[col].hist()
    plt.show()

data_loader.preprocess_data()
data_loader.data.head()
