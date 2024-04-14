from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/content/drive')

# Load the training dataset from Google Drive
train = pd.read_csv('/content/drive/My Drive/train.csv')

train.head(5)

def clean_currency(x):
    if isinstance(x, str):
        if 'Crore+' in x:
            return float(x.replace(' Crore+', '')) * 10000000
        elif 'Lac+' in x:
            return float(x.replace(' Lac+', '')) * 100000
        elif 'Thou+' in x:
            return float(x.replace(' Thou+', '')) * 1000
        elif 'Hund+' in x:
            return float(x.replace(' Hund+', '')) * 1000
        else:
            return float(x.replace('$', '').replace(',', ''))
    return x

train['Total Assets'] = train['Total Assets'].apply(clean_currency)
train['Liabilities'] = train['Liabilities'].apply(clean_currency)

train.head()

# Calculate the percentage distribution of parties with candidates having the most criminal records
criminal_party_counts = train[train['Criminal Case'] > 2]['Party'].value_counts()
criminal_party_percentages = (criminal_party_counts / criminal_party_counts.sum()) * 100

# Calculate the mean net worth of candidates by party
wealthy_party_means = train.groupby('Party')['Total Assets'].mean().sort_values(ascending=False).head(10)

# Plot the results
plt.figure(figsize=(14, 6))

# Plot the percentage distribution of parties with criminal records
plt.figure(figsize=(12, 6))  # Adjust the width as needed

plt.subplot(1, 2, 1)
criminal_party_percentages.plot(kind='bar', color='skyblue')
plt.title('Percentage Distribution of Parties with Criminal Records')
plt.xlabel('Party')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Find parties with less than 1.5% criminal records
small_parties = criminal_party_percentages[criminal_party_percentages < 1.5]
# Combine them into an "Others" category
others_percentage = small_parties.sum()
criminal_party_percentages = criminal_party_percentages[criminal_party_percentages >= 1.5]
criminal_party_percentages['Others'] = others_percentage

# Draw pie chart for criminal records with "Others" category
plt.figure(figsize=(15, 8))  # Adjust the width as needed
plt.subplot(1, 2, 2)
plt.pie(criminal_party_percentages, labels=criminal_party_percentages.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage Distribution of Parties with Criminal Records')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()

plt.show()

# Plot the percentage distribution of parties with the most wealthy candidates
plt.figure(figsize=(12, 6))  # Adjust the width as needed

plt.subplot(1, 2, 2)
wealthy_party_means.plot(kind='bar', color='lightgreen')
plt.title('Candidates by Party')
plt.xlabel('Party')
plt.ylabel('Mean Net Worth')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

plt.figure(figsize=(15, 8))  # Adjust the width as needed
plt.subplot(1, 2, 2)
plt.pie(wealthy_party_means, labels=wealthy_party_means.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage Distribution of Parties with the Most Wealthy Candidates')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()

plt.show()
