import os
from nicegui import ui, app
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure the application
app.title = "ML Engineer Portfolio"
app.favicon = "🧠"

# Define color scheme
PRIMARY_COLOR = "#4F46E5"  # Indigo
SECONDARY_COLOR = "#10B981"  # Emerald
BG_COLOR = "#F9FAFB"
TEXT_COLOR = "#1F2937"
CARD_BG = "#FFFFFF"

# Sample data for the portfolio
class PortfolioData:
    def __init__(self):
        self.personal_info = {
            "name": "Alex Johnson",
            "title": "Machine Learning Engineer",
            "bio": "Passionate ML Engineer with 5+ years of experience developing and deploying machine learning models. Specialized in computer vision and NLP with a strong foundation in Python and data science.",
            "location": "San Francisco, CA",
            "email": "alex@example.com",
            "github": "github.com/alexjohnson",
            "linkedin": "linkedin.com/in/alexjohnson",
            "profile_image": "/static/profile.jpg"  # This will be a placeholder
        }
        
        self.skills = {
            "Machine Learning": 95,
            "Deep Learning": 90,
            "Python": 95,
            "TensorFlow/PyTorch": 85,
            "Computer Vision": 80,
            "NLP": 85,
            "Data Analysis": 90,
            "MLOps": 75,
            "SQL": 80,
            "Cloud Platforms": 85
        }
        
        self.projects = [
            {
                "title": "Image Classification System",
                "description": "Developed a convolutional neural network for classifying medical images with 94% accuracy, improving diagnostic speed by 60%.",
                "technologies": ["PyTorch", "Python", "Docker", "AWS"],
                "image": "project1.jpg",
                "demo_type": "classification"
            },
            {
                "title": "NLP Sentiment Analysis",
                "description": "Created a BERT-based sentiment analysis model for customer feedback, achieving 92% accuracy and reducing manual review time by 75%.",
                "technologies": ["TensorFlow", "Hugging Face", "Python", "GCP"],
                "image": "project2.jpg",
                "demo_type": "none"
            },
            {
                "title": "Predictive Maintenance System",
                "description": "Built an end-to-end system that predicts equipment failures 48 hours in advance, reducing downtime by 35% and saving $2M annually.",
                "technologies": ["Scikit-learn", "PySpark", "MLflow", "Azure"],
                "image": "project3.jpg",
                "demo_type": "timeseries"
            },
            {
                "title": "Recommendation Engine",
                "description": "Implemented a hybrid recommendation system that increased user engagement by 28% and purchase conversion by 15%.",
                "technologies": ["Python", "TensorFlow", "Neo4j", "AWS"],
                "image": "project4.jpg",
                "demo_type": "none"
            }
        ]
        
        self.experience = [
            {
                "role": "Senior ML Engineer",
                "company": "TechInnovate AI",
                "period": "2020 - Present",
                "description": "Lead ML engineer for computer vision projects, managing a team of 5 engineers and delivering solutions that increased revenue by 40%."
            },
            {
                "role": "Machine Learning Engineer",
                "company": "DataSmart Solutions",
                "period": "2018 - 2020",
                "description": "Developed and deployed NLP models for text classification and entity extraction, processing over 10M documents monthly."
            },
            {
                "role": "Data Scientist",
                "company": "AnalyticsHub",
                "period": "2016 - 2018",
                "description": "Created predictive models for customer churn and lifetime value, improving retention strategies by 25%."
            }
        ]
        
        self.education = [
            {
                "degree": "M.S. in Computer Science, AI Specialization",
                "institution": "Stanford University",
                "year": "2016"
            },
            {
                "degree": "B.S. in Mathematics and Statistics",
                "institution": "University of California, Berkeley",
                "year": "2014"
            }
        ]

# Initialize portfolio data
portfolio = PortfolioData()

# Create a placeholder profile image if it doesn't exist
def ensure_static_dir():
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    # Create a placeholder profile image
    profile_path = os.path.join(static_dir, 'profile.jpg')
    if not os.path.exists(profile_path):
        # This is just a comment - in a real app, we'd create a placeholder image
        # For now, we'll use a UI avatar in the code instead
        pass

ensure_static_dir()

# ML Demo Utilities
class MLDemos:
    @staticmethod
    def generate_classification_demo():
        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000, n_features=2, n_informative=2, n_redundant=0, 
            n_clusters_per_class=1, random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Create visualization
        df = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2'])
        df['Predicted Class'] = y_pred
        
        fig = px.scatter(
            df, x='Feature 1', y='Feature 2', color='Predicted Class',
            title=f'Classification Results (Accuracy: {accuracy:.2%})',
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR]
        )
        
        # Add decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig_boundary = go.Figure(fig)
        fig_boundary.add_trace(
            go.Contour(
                z=Z, x=xx[0], y=yy[:, 0], showscale=False,
                colorscale=[[0, 'rgba(79, 70, 229, 0.1)'], [1, 'rgba(16, 185, 129, 0.1)']],
                line=dict(width=0),
                contours=dict(showlabels=False)
            )
        )
        
        fig_boundary.update_layout(
            plot_bgcolor='white',
            width=600,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        return fig_boundary
    
    @staticmethod
    def generate_timeseries_demo():
        # Generate synthetic time series data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
        baseline = np.sin(np.linspace(0, 2*np.pi*2, len(dates))) * 10 + 50
        noise = np.random.normal(0, 1, len(dates))
        trend = np.linspace(0, 15, len(dates))
        values = baseline + noise + trend
        
        # Create anomalies
        anomaly_indices = [60, 120, 180, 250, 300]
        anomalies = np.zeros(len(dates))
        for idx in anomaly_indices:
            values[idx] += 15 if np.random.random() > 0.5 else -15
            anomalies[idx] = 1
        
        # Create dataframe
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'is_anomaly': anomalies
        })
        
        # Train a simple model to predict the next 30 days
        train_df = df[:-30].copy()
        test_df = df[-30:].copy()
        
        # Create features (simple lag features)
        for lag in [1, 7, 14]:
            train_df[f'lag_{lag}'] = train_df['value'].shift(lag)
        
        train_df = train_df.dropna()
        
        # Train model
        X_train = train_df[[f'lag_{lag}' for lag in [1, 7, 14]]]
        y_train = train_df['value']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Create visualization
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['value'],
            mode='lines',
            name='Equipment Readings',
            line=dict(color=PRIMARY_COLOR, width=2)
        ))
        
        # Add anomalies
        anomaly_df = df[df['is_anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=anomaly_df['date'], y=anomaly_df['value'],
            mode='markers',
            name='Detected Anomalies',
            marker=dict(color='red', size=10, symbol='circle')
        ))
        
        # Add prediction range
        prediction_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        prediction_values = values[-1] + np.linspace(1, 5, 30) + np.sin(np.linspace(0, np.pi, 30)) * 5
        
        fig.add_trace(go.Scatter(
            x=prediction_dates, y=prediction_values,
            mode='lines',
            name='Predicted Values',
            line=dict(color=SECONDARY_COLOR, width=2, dash='dash')
        ))
        
        # Add confidence interval
        upper_bound = prediction_values + 5
        lower_bound = prediction_values - 5
        
        fig.add_trace(go.Scatter(
            x=prediction_dates, y=upper_bound,
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=prediction_dates, y=lower_bound,
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(16, 185, 129, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Predictive Maintenance: Equipment Readings & Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Sensor Reading',
            plot_bgcolor='white',
            width=700,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig

# UI Components
class PortfolioUI:
    def __init__(self):
        self.dark_mode = False
        self.current_page = 'home'
        
    def setup_ui(self):
        # Apply global styles
        ui.add_head_html('''
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            body {
                font-family: 'Inter', sans-serif;
                background-color: #F9FAFB;
                color: #1F2937;
            }
            .n-card {
                border-radius: 8px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }
            .skill-bar {
                height: 8px;
                border-radius: 4px;
                background-color: #E5E7EB;
                overflow: hidden;
                margin-bottom: 16px;
            }
            .skill-progress {
                height: 100%;
                border-radius: 4px;
                background-color: #4F46E5;
            }
            .nav-link {
                padding: 8px 16px;
                border-radius: 6px;
                transition: all 0.2s;
                font-weight: 500;
            }
            .nav-link:hover {
                background-color: rgba(79, 70, 229, 0.1);
            }
            .nav-link.active {
                background-color: #4F46E5;
                color: white;
            }
            .project-card {
                transition: transform 0.2s;
            }
            .project-card:hover {
                transform: translateY(-5px);
            }
            .tech-tag {
                background-color: rgba(79, 70, 229, 0.1);
                color: #4F46E5;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: 500;
            }
        </style>
        ''')
        
        # Create layout
        with ui.header().classes('flex justify-between items-center p-4 bg-white shadow-sm'):
            ui.label('Alex Johnson').classes('text-xl font-bold text-indigo-600')
            
            with ui.row().classes('gap-2'):
                ui.button('Home', on_click=lambda: self.navigate('home')).classes('nav-link')
                ui.button('Projects', on_click=lambda: self.navigate('projects')).classes('nav-link')
                ui.button('Skills', on_click=lambda: self.navigate('skills')).classes('nav-link')
                ui.button('Experience', on_click=lambda: self.navigate('experience')).classes('nav-link')
                ui.button('Contact', on_click=lambda: self.navigate('contact')).classes('nav-link')
        
        # Create pages container
        self.pages_container = ui.element('div').classes('p-4 max-w-7xl mx-auto')
        
        # Create pages
        self.create_home_page()
        self.create_projects_page()
        self.create_skills_page()
        self.create_experience_page()
        self.create_contact_page()
        
        # Show default page
        self.navigate('home')
    
    def navigate(self, page_name):
        self.current_page = page_name
        for page in self.pages_container.children:
            page.visible = False
        
        for page in self.pages_container.children:
            if page.id == page_name:
                page.visible = True
                break
    
    def create_home_page(self):
        with self.pages_container:
            with ui.element('div').classes('py-8').id('home'):
                with ui.row().classes('items-center flex-col md:flex-row gap-8'):
                    with ui.card().classes('w-full md:w-1/3'):
                        ui.image('https://ui-avatars.com/api/?name=Alex+Johnson&background=4F46E5&color=fff&size=256').classes('w-full rounded-lg')
                    
                    with ui.card().classes('w-full md:w-2/3 p-6'):
                        ui.label('Hello, I\'m').classes('text-lg text-gray-600')
                        ui.label(portfolio.personal_info['name']).classes('text-4xl font-bold text-gray-900')
                        ui.label(portfolio.personal_info['title']).classes('text-xl text-indigo-600 font-medium mb-4')
                        
                        ui.label(portfolio.personal_info['bio']).classes('text-gray-700 mb-6')
                        
                        with ui.row().classes('gap-4'):
                            ui.button('View Projects', on_click=lambda: self.navigate('projects')).classes('bg-indigo-600 text-white')
                            ui.button('Contact Me', on_click=lambda: self.navigate('contact')).classes('border border-indigo-600 text-indigo-600')
                
                ui.separator().classes('my-8')
                
                with ui.card().classes('p-6 mt-8'):
                    ui.label('Featured Project: Predictive Maintenance System').classes('text-2xl font-bold mb-4')
                    ui.label('Interactive Demo').classes('text-lg font-medium text-indigo-600 mb-2')
                    
                    # Add interactive demo
                    demo_fig = MLDemos.generate_timeseries_demo()
                    ui.plotly(demo_fig).classes('w-full')
                    
                    ui.label('This system predicts equipment failures 48 hours in advance, reducing downtime by 35%.').classes('mt-4 text-gray-700')
    
    def create_projects_page(self):
        with self.pages_container:
            with ui.element('div').classes('py-8').id('projects').style('display: none'):
                ui.label('My Projects').classes('text-3xl font-bold mb-8')
                
                with ui.grid(columns=2).classes('gap-6'):
                    for project in portfolio.projects:
                        with ui.card().classes('project-card'):
                            # Use a placeholder image with project title
                            img_url = f"https://ui-avatars.com/api/?name={project['title'].replace(' ', '+')}&background=4F46E5&color=fff&size=256"
                            ui.image(img_url).classes('w-full h-48 object-cover rounded-t-lg')
                            
                            with ui.card_section().classes('p-4'):
                                ui.label(project['title']).classes('text-xl font-bold')
                                ui.label(project['description']).classes('text-gray-700 my-2')
                                
                                with ui.row().classes('flex-wrap gap-2 mt-3'):
                                    for tech in project['technologies']:
                                        ui.label(tech).classes('tech-tag')
                                
                                ui.separator().classes('my-4')
                                
                                if project['demo_type'] == 'classification':
                                    ui.label('Interactive Demo').classes('text-lg font-medium text-indigo-600 mb-2')
                                    demo_fig = MLDemos.generate_classification_demo()
                                    ui.plotly(demo_fig).classes('w-full')
                                
                                elif project['demo_type'] == 'timeseries':
                                    ui.label('Interactive Demo').classes('text-lg font-medium text-indigo-600 mb-2')
                                    demo_fig = MLDemos.generate_timeseries_demo()
                                    ui.plotly(demo_fig).classes('w-full')
    
    def create_skills_page(self):
        with self.pages_container:
            with ui.element('div').classes('py-8').id('skills').style('display: none'):
                ui.label('My Skills').classes('text-3xl font-bold mb-8')
                
                with ui.row().classes('gap-8'):
                    with ui.card().classes('w-full md:w-1/2 p-6'):
                        ui.label('Technical Skills').classes('text-xl font-bold mb-4')
                        
                        for skill, level in portfolio.skills.items():
                            ui.label(f"{skill} ({level}%)").classes('font-medium')
                            with ui.element('div').classes('skill-bar'):
                                ui.element('div').classes('skill-progress').style(f'width: {level}%')
                    
                    with ui.card().classes('w-full md:w-1/2 p-6'):
                        ui.label('Skills Visualization').classes('text-xl font-bold mb-4')
                        
                        # Create radar chart for skills
                        skills = list(portfolio.skills.keys())
                        values = list(portfolio.skills.values())
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=skills,
                            fill='toself',
                            fillcolor=f'rgba({PRIMARY_COLOR.lstrip("#")}, 0.2)',
                            line=dict(color=PRIMARY_COLOR)
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )
                            ),
                            showlegend=False,
                            width=500,
                            height=500
                        )
                        
                        ui.plotly(fig)
    
    def create_experience_page(self):
        with self.pages_container:
            with ui.element('div').classes('py-8').id('experience').style('display: none'):
                ui.label('Experience & Education').classes('text-3xl font-bold mb-8')
                
                with ui.row().classes('gap-8'):
                    with ui.card().classes('w-full md:w-1/2 p-6'):
                        ui.label('Work Experience').classes('text-xl font-bold mb-4')
                        
                        for job in portfolio.experience:
                            with ui.element('div').classes('mb-6'):
                                ui.label(job['role']).classes('text-lg font-bold')
                                with ui.row().classes('justify-between'):
                                    ui.label(job['company']).classes('text-indigo-600')
                                    ui.label(job['period']).classes('text-gray-600')
                                ui.label(job['description']).classes('text-gray-700 mt-2')
                    
                    with ui.card().classes('w-full md:w-1/2 p-6'):
                        ui.label('Education').classes('text-xl font-bold mb-4')
                        
                        for edu in portfolio.education:
                            with ui.element('div').classes('mb-6'):
                                ui.label(edu['degree']).classes('text-lg font-bold')
                                with ui.row().classes('justify-between'):
                                    ui.label(edu['institution']).classes('text-indigo-600')
                                    ui.label(edu['year']).classes('text-gray-600')
    
    def create_contact_page(self):
        with self.pages_container:
            with ui.element('div').classes('py-8').id('contact').style('display: none'):
                ui.label('Contact Me').classes('text-3xl font-bold mb-8')
                
                with ui.row().classes('gap-8'):
                    with ui.card().classes('w-full md:w-1/2 p-6'):
                        ui.label('Get In Touch').classes('text-xl font-bold mb-4')
                        
                        with ui.element('div').classes('space-y-4'):
                            ui.label(f"📧 Email: {portfolio.personal_info['email']}").classes('text-gray-700')
                            ui.label(f"📍 Location: {portfolio.personal_info['location']}").classes('text-gray-700')
                            ui.label(f"🔗 GitHub: {portfolio.personal_info['github']}").classes('text-gray-700')
                            ui.label(f"🔗 LinkedIn: {portfolio.personal_info['linkedin']}").classes('text-gray-700')
                    
                    with ui.card().classes('w-full md:w-1/2 p-6'):
                        ui.label('Send Me a Message').classes('text-xl font-bold mb-4')
                        
                        ui.input(label='Name').classes('w-full mb-4')
                        ui.input(label='Email').classes('w-full mb-4')
                        ui.input(label='Subject').classes('w-full mb-4')
                        ui.textarea(label='Message').classes('w-full mb-4').style('min-height: 120px')
                        
                        ui.button('Send Message', on_click=lambda: ui.notify('Message sent! (This is a demo)')).classes('bg-indigo-600 text-white')

# Initialize the application
def init():
    portfolio_ui = PortfolioUI()
    portfolio_ui.setup_ui()
    return app

# Create the application instance
application = init()