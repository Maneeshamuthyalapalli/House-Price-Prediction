df['price_per_sqft'] = df['price'] / df['area']
df['price_per_sqft'] = df['price'] / df['area']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['room_to_area_ratio'] = df['total_rooms'] / df['area']
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
from sklearn.preprocessing import StandardScaler
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price_per_sqft', 'total_rooms', 'room_to_area_ratio']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print(df.head())
