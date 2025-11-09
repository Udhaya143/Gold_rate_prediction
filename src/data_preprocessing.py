from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values (fill with the mean)
    df.fillna(df.mean(), inplace=True)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Split the data into training and testing sets
    X = scaled_data[:, :-1]  # Features (all columns except the last)
    y = scaled_data[:, -1]   # Target (last column)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler
