from flask import Flask, render_template, request
import pandas as pd
import os, sys
from recommender import HybridRecommender

app = Flask(__name__)

# forces Flask to locate the templates folder correctly:
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
app = Flask(__name__, template_folder=template_dir)

print(f"\nFlask is looking for HTML layout at: {template_dir}")

if not os.path.exists(template_dir):
    os.makedirs(template_dir)
    print(f"Created missing folder automatically: {template_dir}")

html_file_path = os.path.join(template_dir, 'index.html')

# Always write/refresh the premium interface file
print(f"Deploying cinematic user interface design...")
# with open(html_file_path, 'w', encoding='utf-8') as f

# --- Data Preparation & Model Initialization ---
csv_path = 'TMDB_movie_dataset_v11.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing dataset at {csv_path}. Please check your folders.")

print("Loading dataset for Flask Server...")
# Loading 30,000 rows to ensure fast startup times and responsive searches
tmdb_df = pd.read_csv(csv_path, low_memory=False).head(30000)

# Pre-cleaning data values
tmdb_df['vote_average'] = pd.to_numeric(tmdb_df['vote_average'], errors='coerce').fillna(0)
tmdb_df['vote_count'] = pd.to_numeric(tmdb_df['vote_count'], errors='coerce').fillna(0)
tmdb_df['popularity'] = pd.to_numeric(tmdb_df['popularity'], errors='coerce').fillna(0)
tmdb_df = tmdb_df.dropna(subset=['title']).reset_index(drop=True)

# Instantiating and training your custom engine
recommender = HybridRecommender(df=tmdb_df)
recommender.train_content_based()
recommender.train_collaborative_features()
print("Flask Engine is Ready for Requests!")


# --- Web Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    searched_movie = None
    error_message = None

    if request.method == 'POST':
        searched_movie = request.form.get('movie_name', '').strip()
        
        if searched_movie:
            try:
                # Call your engine without needing a user_id variable
                results_df = recommender.get_hybrid_recommendations(movie_title=searched_movie, top_n=10)
                # Convert results to a dictionary list format for Jinja templating
                recommendations = results_df.to_dict('records')
            except KeyError:
                error_message = f"'{searched_movie}' was not found in our database subset. Try searching for another movie!"
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"

    return render_template('index.html', recs=recommendations, movie_name=searched_movie, error=error_message)

if __name__ == '__main__':
    # Runs the Flask local server
    app.run(debug=True)