import streamlit as st
import re
from knn_model import load_data, train_knn, predict_knn

def normalize_text(text):
    """Normalize text by removing special characters and converting to lowercase."""
    text = re.sub(r'[\s/]+', ' ', text)  # Replace slashes and multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip().lower()

# Load the data and prepare the model
df, reverse_mappings = load_data("job.csv")
features = ['Industry', 'Role']  # The features used in the KNN
target = 'Role Category'
knn_model = train_knn(df, features, target)

# Normalize industry and role names in reverse_mappings to ensure consistency
reverse_mappings['Industry'] = {k: normalize_text(v) for k, v in reverse_mappings['Industry'].items()}
reverse_mappings['Role'] = {k: normalize_text(v) for k, v in reverse_mappings['Role'].items()}

# User Interface
st.title('Job Recommendation System')

# Default select option
user_ids = ['Select a user ID'] + df['Uniq Id'].tolist()
user_id = st.selectbox('Please select user ID to recommend:', user_ids, index=0)

# Process selection
if user_id != 'Select a user ID':
    selected_user = df[df['Uniq Id'] == user_id]

    if not selected_user.empty:
        st.markdown('### Selected User ID Information', unsafe_allow_html=True)
        # Display selected user ID information in a friendly format with yellow color
        info_columns = ["Job Title", "Key Skills", "Role Category", "Location", "Functional Area", "Industry"]
        for col in info_columns:
            value = reverse_mappings[col][selected_user[col].iloc[0]] if col in reverse_mappings else selected_user[col].iloc[0]
            st.markdown(f"<span style='color: lightgreen;'>{col}: {value}</span>", unsafe_allow_html=True)

        # Button to trigger recommendations
        if st.button('Recommend'):
            # Get recommendations
            user_features = selected_user[features].iloc[0].tolist()
            user_pred = predict_knn(knn_model, user_features)
            
            # Fetch recommendations, group by industry and aggregate roles
            recommendations = df[df[target] == user_pred][['Industry', 'Role']].drop_duplicates().head(12)
            recommendations['Industry'] = recommendations['Industry'].apply(lambda x: normalize_text(reverse_mappings['Industry'][x]))
            recommendations['Role'] = recommendations['Role'].apply(lambda x: normalize_text(reverse_mappings['Role'][x]))
            grouped_recommendations = recommendations.groupby('Industry')['Role'].agg(lambda x: ', '.join(set(x))).reset_index()

            # Display recommendations
            st.markdown('### Recommendations based on selected ID', unsafe_allow_html=True)
            count = 1
            for index, row in grouped_recommendations.iterrows():
                industry = row['Industry']
                roles = row['Role']
                st.markdown(f"{count}. <span style='color: pink;'>Industry: {industry}</span>", unsafe_allow_html=True)
                st.markdown(f"   <span style='color: cyan;'>Role: {roles}</span>", unsafe_allow_html=True)
                count += 1










