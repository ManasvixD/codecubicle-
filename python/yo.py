import pandas as pd
from sqlalchemy import create_engine
from transformers import pipeline
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Establish the SQLAlchemy connection
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Koki9205',
    'database': 'data'
}

# Connection string using SQLAlchemy and PyMySQL
connection_str = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:3306/{db_config['database']}"
engine = create_engine(connection_str)

# Step 2: Fetch the data using SQLAlchemy into a pandas DataFrame
query = "SELECT * FROM diabetes_data"
dataframe = pd.read_sql(query, con=engine)

# Step 3: Perform basic statistical analysis
correlations = dataframe.corr()['Outcome'].sort_values(ascending=False)
mean_values = dataframe.mean()
median_values = dataframe.median()

# Step 4: Generate narrative text
narrative = []
for feature in correlations.index[1:]:  # Skip 'Outcome' itself
    correlation = correlations[feature]
    mean = mean_values[feature]
    median = median_values[feature]
    
    if correlation > 0:
        narrative.append(f"As the {feature} increases, the likelihood of being diabetic also increases. "
                         f"The average value of {feature} in the dataset is {mean:.2f}, with a median of {median:.2f}.")
    else:
        narrative.append(f"As the {feature} increases, the likelihood of being diabetic decreases. "
                         f"The average value of {feature} in the dataset is {mean:.2f}, with a median of {median:.2f}.")

# Join narratives into a single text block
narrative_text = "\n".join(narrative)

# Step 5: Format the data into a text block for summarization
data_text = dataframe.head(20).to_string()

# Split the text into smaller chunks that fit within the model's token limit
def split_text(text, chunk_size=1024):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i + chunk_size])

# Step 6: Use Hugging Face's Transformers to summarize the text
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Step 7: Summarize each chunk separately
summaries = []
for chunk in split_text(data_text):
    summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
    summaries.append(summary[0]['summary_text'])

# Join all the summaries
final_summary = " ".join(summaries)

# Output the final summary
print("Summary of the diabetes_data table:")
print(final_summary)

# Step 8: Load the SpaCy English model for NER
nlp = spacy.load("en_core_web_sm")

# Step 9: Process the extracted text with SpaCy for Named Entity Recognition
doc = nlp(narrative_text)

# Step 10: Print the Named Entities, their labels, and corresponding text
print("\nNamed Entities and their Labels:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Step 11: Additional Insights: Parts of Speech and Dependency Parsing
print("\nToken Analysis (Parts of Speech and Dependencies):")
for token in doc:
    print(f"{token.text}: {token.pos_}, {token.dep_}")

# Output the narrative insights
print("\nNarrative Insights:")
print(narrative_text)

# Step 12: Data Visualization with Matplotlib
# Set the style for seaborn
sns.set(style="whitegrid")

# Histogram of Glucose levels
plt.figure(figsize=(10, 6))
sns.histplot(dataframe['Glucose'], bins=30, kde=True)
plt.title('Distribution of Glucose Levels')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Glucose levels vs Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outcome', y='Glucose', data=dataframe)
plt.title('Glucose Levels by Diabetes Outcome')
plt.xlabel('Diabetes Outcome')
plt.ylabel('Glucose Level')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation = dataframe.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Diabetes Data')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(dataframe, hue='Outcome')
plt.title('Pairplot of Features Colored by Diabetes Outcome')
plt.show()

# Step 13: Plot each feature against 'DiabetesPedigreeFunction'
columns_to_plot = [col for col in dataframe.columns if col != 'DiabetesPedigreeFunction']

# Plot each feature against 'DiabetesPedigreeFunction'
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    
    if dataframe[column].dtype in ['int64', 'float64']:  # If the column is numerical
        # Scatter plot for numerical columns
        sns.scatterplot(x='DiabetesPedigreeFunction', y=column, data=dataframe)
        plt.title(f'{column} vs DiabetesPedigreeFunction')
        plt.xlabel('DiabetesPedigreeFunction')
        plt.ylabel(column)
        
    else:
        # Box plot for categorical columns like 'Outcome'
        sns.boxplot(x=column, y='DiabetesPedigreeFunction', data=dataframe)
        plt.title(f'{column} vs DiabetesPedigreeFunction')
        plt.xlabel(column)
        plt.ylabel('DiabetesPedigreeFunction')
    
    plt.show()
