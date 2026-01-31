"""
Spotify Analytics Dashboard - Real Data Edition
Music Track Analytics with Real Spotify Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from sklearn.ensemble import RandomForestRegressor

# ==================== PAGE CONFIG (MUST BE FIRST LINE) ====================
st.set_page_config(
    page_title="Spotify Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MINIMAL SAFE CSS ====================
st.markdown("""
<style>
    /* Dark Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Card Styling for Metrics */
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        border: 1px solid #2b2f3d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #161720;
    }
    
    /* Metric text colors */
    [data-testid="stMetricLabel"] {
        color: #9CA3AF;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    
    /* Headers white */
    h1, h2, h3 {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def load_india_geojson():
    """Load India states GeoJSON for choropleth map"""
    url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    response = requests.get(url)
    return response.json()

# Language to State mapping
lang_to_state = {
    'Punjabi': 'Punjab',
    'Hindi': 'Delhi',
    'Marathi': 'Maharashtra',
    'Tamil': 'Tamil Nadu',
    'Telugu': 'Telangana',
    'Malayalam': 'Kerala',
    'Kannada': 'Karnataka',
    'Gujarati': 'Gujarat',
    'Bengali': 'West Bengal',
    'Odia': 'Odisha',
    'Assamese': 'Assam'
}

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    df = pd.read_csv('spotify_cleaned_master.csv')
    return df

# Load data
df = load_data()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("🎵 Spotify Analytics")
    st.markdown("---")
    
    # Navigation
    selected_tab = st.radio(
        'Navigate',
        ['Executive Overview', 'Hit DNA', 'Athena AI'],
        label_visibility='collapsed'
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ Filters")
    
    # Language Filter
    selected_languages = st.multiselect(
        'Select Languages',
        df['language'].unique(),
        default=df['language'].unique()[:2]
    )
    
    # Year Filter
    selected_year = st.slider(
        'Select Year Range',
        int(df['Year'].min()),
        int(df['Year'].max()),
        (2015, 2023)
    )

# ==================== APPLY FILTERS ====================
filtered_df = df[
    (df['language'].isin(selected_languages)) &
    (df['Year'] >= selected_year[0]) &
    (df['Year'] <= selected_year[1])
]

# ==================== MAIN CONTENT ====================
st.title("🎧 Welcome to Spotify Analytics")
st.markdown("---")

# ==================== EXECUTIVE OVERVIEW ====================
if selected_tab == 'Executive Overview':
    
    # TOP ROW: 3 METRICS
    col1, col2, col3 = st.columns(3)
    
    total_tracks = len(filtered_df)
    avg_popularity = filtered_df['popularity'].mean()
    
    # Calculate Dominant Language by Total Stream Count
    stream_by_lang = filtered_df.groupby('language')['Stream'].sum().sort_values(ascending=False)
    dominant_language = stream_by_lang.index[0] if len(stream_by_lang) > 0 else "N/A"
    dominant_streams = stream_by_lang.iloc[0] if len(stream_by_lang) > 0 else 0
    
    # Format streams as M/B
    if dominant_streams >= 1_000_000_000:
        stream_text = f"{dominant_streams / 1_000_000_000:.2f}B Streams"
    elif dominant_streams >= 1_000_000:
        stream_text = f"{dominant_streams / 1_000_000:.1f}M Streams"
    else:
        stream_text = f"{dominant_streams:,.0f} Streams"
    
    col1.metric("🎵 Total Tracks", f"{total_tracks:,}")
    col2.metric("⭐ Avg Popularity", f"{avg_popularity:.1f}")
    col3.metric("🏆 Dominant Language", dominant_language, delta=stream_text)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # MIDDLE ROW: CHARTS (1:1 split)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Songs by Language")
        
        # Bar Chart: Count of Songs by Language
        song_count = filtered_df.groupby('language').size().reset_index(name='Count')
        song_count = song_count.sort_values('Count', ascending=False)
        
        fig_bar = px.bar(
            song_count,
            x='language',
            y='Count',
            color='language',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'language': 'Language', 'Count': 'Number of Songs'}
        )
        
        fig_bar.update_layout(
            height=350,
            plot_bgcolor='#1e2130',
            paper_bgcolor='#0E1117',
            xaxis=dict(showgrid=False, color='#9CA3AF'),
            yaxis=dict(showgrid=True, gridcolor='#374151', color='#9CA3AF'),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=40)
        )
        
        st.plotly_chart(fig_bar, width='stretch')
    
    with col_right:
        st.markdown("### 📈 Popularity Over Time")
        
        # Line Chart: Avg Popularity over Year by Language
        popularity_trend = filtered_df.groupby(['Year', 'language'])['popularity'].mean().reset_index()
        
        fig_line = px.line(
            popularity_trend,
            x='Year',
            y='popularity',
            color='language',
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={'Year': 'Year', 'popularity': 'Average Popularity', 'language': 'Language'}
        )
        
        fig_line.update_layout(
            height=350,
            plot_bgcolor='#1e2130',
            paper_bgcolor='#0E1117',
            xaxis=dict(showgrid=True, gridcolor='#374151', color='#9CA3AF'),
            yaxis=dict(showgrid=True, gridcolor='#374151', color='#9CA3AF'),
            legend=dict(font=dict(color='#9CA3AF')),
            margin=dict(l=20, r=20, t=20, b=40)
        )
        
        st.plotly_chart(fig_line, width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # BOTTOM ROW: INDIA CHOROPLETH MAP
    st.markdown("### 🗺️ Music Popularity Across India")
    
    # Load GeoJSON
    india_geojson = load_india_geojson()
    
    # Extract ALL state names from GeoJSON to create canvas
    all_states = [feature['properties']['ST_NM'] for feature in india_geojson['features']]
    canvas_df = pd.DataFrame({'State': all_states})
    
    # Prepare data: Group by language and aggregate
    lang_stats = filtered_df.groupby('language').agg({
        'popularity': 'mean',
        'song_name': 'count'
    }).reset_index()
    lang_stats.rename(columns={'song_name': 'track_count'}, inplace=True)
    
    # Map language to State
    lang_stats['State'] = lang_stats['language'].map(lang_to_state)
    lang_stats = lang_stats.dropna(subset=['State'])
    
    # LEFT MERGE: Canvas (all states) with data (selected states only)
    map_data = canvas_df.merge(lang_stats[['State', 'language', 'popularity', 'track_count']], 
                                on='State', how='left')
    
    # Fill NaN with 0 so ALL states render  
    map_data['popularity_display'] = map_data['popularity'].fillna(0)
    map_data['has_data'] = map_data['popularity'].notna()
    
    # Get actual data range
    actual_values = map_data[map_data['has_data']]['popularity']
    if len(actual_values) > 0:
        v_min = actual_values.min()
        v_max = actual_values.max()
    else:
        v_min, v_max = 0, 100
    
    # Create choropleth with go.Choropleth
    fig_map = go.Figure(go.Choropleth(
        geojson=india_geojson,
        featureidkey='properties.ST_NM',
        locations=map_data['State'],
        z=map_data['popularity_display'],
        zmin=v_min,  # Start from actual min, not 0
        zmax=v_max,  # End at actual max
        colorscale='Viridis',
        marker_line_color='#555',
        marker_line_width=1.5,
        showscale=True,
        colorbar=dict(
            title=dict(text="Popularity", font=dict(color='#FFFFFF')),
            tickfont=dict(color='#9CA3AF')
        ),
        hovertemplate='<b>%{location}</b><br>' +
                      'Language: %{customdata[0]}<br>' +
                      'Popularity: %{customdata[1]}<br>' +
                      'Tracks: %{customdata[2]}<extra></extra>',
        customdata=np.column_stack([
            map_data['language'].fillna('No Data'),
            map_data.apply(lambda row: f"{row['popularity']:.1f}" if row['has_data'] else "N/A", axis=1),
            map_data['track_count'].fillna(0).astype(int)
        ])
    ))
    
    # Update geo layout to show full India
    fig_map.update_geos(
        visible=False,
        scope='asia',
        center=dict(lat=22.5, lon=79),
        projection_scale=4.5
    )
    
    fig_map.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='#0E1117',
            landcolor='#1e2130'
        )
    )
    
    st.plotly_chart(fig_map, width='stretch')

# ==================== HIT DNA (UPGRADED WITH ML) ====================
elif selected_tab == 'Hit DNA':
    st.markdown("## 🧬 Hit DNA - Advanced Audio Intelligence")
    st.markdown("*Diagnose, analyze, and predict with machine learning*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== SECTION A: AUDIO FINGERPRINT (RADAR CHART) ==========
    st.markdown("### 📊 Section A: Audio Fingerprint")
    st.markdown("*Compare selected languages against market average baseline*")
    
    # Audio features to analyze
    audio_features = ['danceability', 'energy', 'acousticness', 'liveness']
    
    # Calculate OVERALL MARKET AVERAGE (from full dataset)
    market_avg = df[audio_features].mean().tolist()
    market_avg.append(market_avg[0])  # Close the polygon
    
    # Calculate average audio features by selected language
    language_features = filtered_df.groupby('language')[audio_features].mean()
    
    # Create Radar Chart
    fig_radar = go.Figure()
    
    # TRACE 1: Market Average Baseline (White Dotted Line with Faint Fill)
    fig_radar.add_trace(go.Scatterpolar(
        r=market_avg,
        theta=audio_features + [audio_features[0]],
        fill='toself',
        name='Market Average',
        fillcolor='rgba(255, 255, 255, 0.1)',  # Very faint white fill
        line=dict(color='rgba(255, 255, 255, 0.8)', width=2, dash='dot'),  # White dotted line
        opacity=1
    ))
    
    # TRACE 2: Overlay Selected Languages (Semi-transparent)
    colors = px.colors.qualitative.Vivid
    
    for idx, language in enumerate(language_features.index):
        values = language_features.loc[language].tolist()
        values.append(values[0])  # Close the radar chart
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=audio_features + [audio_features[0]],
            fill='toself',
            name=language,
            line=dict(color=colors[idx % len(colors)], width=3),
            opacity=0.5  # Semi-transparent to see baseline through
        ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#374151',
                color='#9CA3AF'
            ),
            angularaxis=dict(
                gridcolor='#374151',
                color='#9CA3AF'
            )
        ),
        showlegend=True,
        legend=dict(font=dict(color='#9CA3AF'), orientation='h', yanchor='bottom', y=-0.2),
        height=450,
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent
        margin=dict(l=80, r=80, t=20, b=80)
    )
    
    st.plotly_chart(fig_radar, width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== SECTION B: WHAT DRIVES A HIT? (CORRELATION BAR CHART) ==========
    st.markdown("### 🎯 Section B: What Drives a Hit?")
    st.markdown("*Feature impact on popularity - green boosts hits, red hurts hits*")
    
    # Calculate correlation of audio features with popularity
    feature_list = ['danceability', 'energy', 'acousticness', 'liveness']
    if 'duration_sec' in filtered_df.columns:
        feature_list.append('duration_sec')
    
    correlations = {}
    for feature in feature_list:
        corr = filtered_df[['popularity', feature]].corr().iloc[0, 1]
        correlations[feature] = corr
    
    # Create DataFrame for plotting
    corr_df = pd.DataFrame({
        'Feature': [f.capitalize() for f in correlations.keys()],
        'Correlation': list(correlations.values())
    })
    
    # Sort by correlation value
    corr_df = corr_df.sort_values('Correlation', ascending=True)
    
    # Assign colors: Green for positive, Red for negative
    corr_df['Color'] = corr_df['Correlation'].apply(
        lambda x: '#22c55e' if x > 0 else '#ef4444'  # Green or Red
    )
    
    # Create horizontal bar chart
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        y=corr_df['Feature'],
        x=corr_df['Correlation'],
        orientation='h',
        marker=dict(
            color=corr_df['Color'],
            line=dict(color='#555', width=1)
        ),
        text=corr_df['Correlation'].apply(lambda x: f'{x:.3f}'),
        textposition='outside',
        textfont=dict(color='#9CA3AF', size=12),
        hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
    ))
    
    fig_bar.update_layout(
        height=350,
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1e2130',
        font=dict(color='#9CA3AF'),
        xaxis=dict(
            title='Correlation Coefficient',
            range=[-1, 1],
            gridcolor='#374151',
            zerolinecolor='#666',
            zerolinewidth=2,
            tickfont=dict(color='#9CA3AF')
        ),
        yaxis=dict(
            title='',
            tickfont=dict(color='#FFFFFF', size=12)
        ),
        margin=dict(l=120, r=40, t=20, b=60),
        showlegend=False
    )
    
    st.plotly_chart(fig_bar, width='stretch')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== SECTION C: PREDICTIVE ENGINE (ML MODEL) ==========
    st.markdown("### 🤖 Section C: Predictive Engine")
    st.markdown("*AI-powered popularity prediction based on audio features*")
    
    # Train RandomForest model
    @st.cache_resource
    def train_model():
        # Prepare data
        X = df[audio_features].dropna()
        y = df.loc[X.index, 'popularity']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        return model
    
    model = train_model()
    
    # Create UI for user input
    st.markdown("**🎛️ Adjust Audio Parameters:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        danceability_input = st.slider(
            "🕺 Danceability",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.01,
            help="How suitable for dancing (0-1)"
        )
    
    with col2:
        energy_input = st.slider(
            "⚡ Energy",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01,
            help="Intensity and activity (0-1)"
        )
    
    with col3:
        acousticness_input = st.slider(
            "🎸 Acousticness",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Presence of acoustic instruments (0-1)"
        )
    
    with col4:
        liveness_input = st.slider(
            "🎤 Liveness",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Presence of live audience (0-1)"
        )
    
    # Make prediction
    input_features = np.array([[danceability_input, energy_input, acousticness_input, liveness_input]])
    predicted_popularity = model.predict(input_features)[0]
    
    # Display prediction prominently
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create 3 columns for centered display
    pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
    
    with pred_col2:
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; 
                        border-radius: 15px; 
                        text-align: center;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.4);'>
                <h3 style='color: #FFFFFF; margin: 0; font-size: 18px;'>🎯 Predicted Popularity Score</h3>
                <h1 style='color: #FFFFFF; margin: 10px 0 0 0; font-size: 48px; font-weight: bold;'>{predicted_popularity:.1f}</h1>
                <p style='color: #E0E0E0; margin: 5px 0 0 0; font-size: 14px;'>out of 100</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Model insights
    st.markdown("<br>", unsafe_allow_html=True)
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("**📈 Model Performance:**")
        st.info(f"✓ Trained on {len(df):,} tracks\n\n✓ RandomForest with 100 trees\n\n✓ Features: {', '.join(audio_features)}")
    
    with insight_col2:
        st.markdown("**💡 Insights:**")
        # Simple insights based on prediction
        if predicted_popularity > 70:
            st.success("🔥 High hit potential! These audio features align with popular tracks.")
        elif predicted_popularity > 50:
            st.warning("📊 Moderate potential. Consider boosting energy or danceability.")
        else:
            st.error("⚠️ Low predicted popularity. Experiment with different audio characteristics.")

# ==================== ATHENA AI ====================
elif selected_tab == 'Athena AI':
    st.markdown("## 🤖 Athena: Music Intelligence")
    st.markdown("*Natural language search powered by smart keyword matching*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== SUGGESTED QUERIES (CLICKABLE BUTTONS) ==========
    st.markdown("### 💡 Try These Queries:")
    
    col1, col2, col3 = st.columns(3)
    
    suggested_query = None
    
    with col1:
        if st.button("🎵 Top 5 Punjabi songs", use_container_width=True):
            suggested_query = "Top 5 Punjabi songs"
    
    with col2:
        if st.button("⚡ Average energy of Hindi vs Tamil", use_container_width=True):
            suggested_query = "Average energy of Hindi vs Tamil"
    
    with col3:
        if st.button("🎤 Who is the most popular artist?", use_container_width=True):
            suggested_query = "Who is the most popular artist?"
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== CUSTOM QUERY INPUT ==========
    user_query = st.text_input(
        "**Or ask your own question:**",
        value=suggested_query if suggested_query else "",
        placeholder="e.g., 'Best Tamil songs' or 'Compare Hindi vs Punjabi danceability'"
    )
    
    # ========== SMART QUERY PROCESSING FUNCTION ==========
    def process_query(query):
        """
        Smart keyword-based query processor that parses natural language 
        and queries the dataframe directly.
        """
        if not query:
            return None
        
        query_lower = query.lower()
        available_languages = df['language'].unique().tolist()
        
        # CASE 1: TOP/BEST SONGS
        if any(keyword in query_lower for keyword in ['top', 'best', 'show me']):
            import re
            
            # Extract language if mentioned
            selected_lang = None
            for lang in available_languages:
                if lang.lower() in query_lower:
                    selected_lang = lang
                    break
            
            # Extract number (default to 5)
            numbers = re.findall(r'\d+', query)
            n = int(numbers[0]) if numbers else 5
            n = min(n, 20)  # Cap at 20
            
            # TIME FILTERING: Detect decades or specific years
            start_year = None
            end_year = None
            time_period_text = ""
            
            # STEP A: Decade Detection (e.g., "2010s", "1990s")
            decade_match = re.search(r'(\d{4})s', query)
            if decade_match:
                start_year = int(decade_match.group(1))
                end_year = start_year + 9  # 2010s = 2010-2019
                time_period_text = f" from {start_year}-{end_year}"
            
            # STEP B: Specific Year (e.g., "2015")
            elif re.search(r'\b(\d{4})\b', query):
                year_match = re.search(r'\b(\d{4})\b', query)
                start_year = int(year_match.group(1))
                end_year = start_year  # Single year
                time_period_text = f" from {start_year}"
            
            # STEP C: Apply time filter FIRST
            filtered_data = df.copy()
            if start_year is not None and end_year is not None:
                filtered_data = filtered_data[(filtered_data['Year'] >= start_year) & (filtered_data['Year'] <= end_year)]
            
            # Get top songs (after time filtering)
            if selected_lang:
                top_songs = filtered_data[filtered_data['language'] == selected_lang].nlargest(n, 'popularity')[
                    ['song_name', 'singer', 'popularity', 'Year']
                ]
                title = f"🏆 Top {n} {selected_lang} Songs{time_period_text}"
            else:
                top_songs = filtered_data.nlargest(n, 'popularity')[
                    ['song_name', 'singer', 'popularity', 'language', 'Year']
                ]
                title = f"🏆 Top {n} Songs{time_period_text}"
            
            return {'type': 'dataframe', 'title': title, 'data': top_songs}
        
        # CASE 2: COMPARE / VS
        elif any(keyword in query_lower for keyword in ['compare', 'vs', 'versus', 'difference']):
            # Extract 2 languages
            mentioned_langs = [lang for lang in available_languages if lang.lower() in query_lower]
            
            if len(mentioned_langs) >= 2:
                lang1, lang2 = mentioned_langs[0], mentioned_langs[1]
                
                # Extract metric (default to popularity)
                metric = 'popularity'
                metric_map = {
                    'energy': 'energy',
                    'dance': 'danceability',
                    'danceability': 'danceability',
                    'acoustic': 'acousticness',
                    'live': 'liveness',
                    'popularity': 'popularity',
                    'popular': 'popularity'
                }
                
                for keyword, col in metric_map.items():
                    if keyword in query_lower:
                        metric = col
                        break
                
                # Calculate means
                val1 = df[df['language'] == lang1][metric].mean()
                val2 = df[df['language'] == lang2][metric].mean()
                
                # Determine which is higher
                if val1 > val2:
                    comparison = f"**{lang1}** has higher {metric} ({val1:.3f}) than **{lang2}** ({val2:.3f})"
                    diff = val1 - val2
                else:
                    comparison = f"**{lang2}** has higher {metric} ({val2:.3f}) than **{lang1}** ({val1:.3f})"
                    diff = val2 - val1
                
                response = f"""
**{lang1} vs {lang2} - {metric.capitalize()} Comparison**

{comparison}

**Difference:** {diff:.3f} ({(diff / max(val1, val2) * 100):.1f}% variation)

**Detailed Stats:**
- {lang1}: {val1:.3f}
- {lang2}: {val2:.3f}
                """
                
                return {'type': 'text', 'content': response}
            else:
                return {'type': 'text', 'content': "⚠️ Please mention 2 languages to compare (e.g., 'Hindi vs Tamil energy')"}
        
        # CASE 3: ARTIST QUERY
        elif 'artist' in query_lower:
            # Find most popular singer by total popularity
            artist_stats = df.groupby('singer').agg({
                'popularity': 'sum',
                'song_name': 'count'
            }).reset_index()
            artist_stats.columns = ['Artist', 'Total Popularity', 'Track Count']
            artist_stats = artist_stats.sort_values('Total Popularity', ascending=False).head(10)
            
            top_artist = artist_stats.iloc[0]
            
            response = f"""
**🎤 Most Popular Artist**

**{top_artist['Artist']}** is the most popular artist!

**Stats:**
- Total Popularity Score: {top_artist['Total Popularity']:.0f}
- Number of Tracks: {int(top_artist['Track Count'])}
- Average Popularity per Track: {top_artist['Total Popularity'] / top_artist['Track Count']:.1f}

**Top 10 Artists:**
            """
            
            return {'type': 'mixed', 'text': response, 'data': artist_stats}
        
        # CASE 4: GENERAL STATS (FALLBACK)
        else:
            response = f"""
**📊 General Dataset Overview**

**Dataset Size:**
- Total Tracks: {len(df):,}
- Languages: {', '.join(available_languages)}
- Year Range: {df['Year'].min()} - {df['Year'].max()}

**Overall Averages:**
- Popularity: {df['popularity'].mean():.2f}
- Energy: {df['energy'].mean():.3f}
- Danceability: {df['danceability'].mean():.3f}
- Acousticness: {df['acousticness'].mean():.3f}

**Top Language:** {df['language'].value_counts().index[0]} ({df['language'].value_counts().iloc[0]:,} tracks)

💡 **Try asking:**
- "Top 5 Hindi songs"
- "Compare Punjabi vs Tamil danceability"
- "Who is the most popular artist?"
            """
            
            return {'type': 'text', 'content': response}
    
    # ========== PROCESS AND DISPLAY RESPONSE ==========
    if user_query:
        st.markdown("---")
        
        # Process the query
        result = process_query(user_query)
        
        if result:
            # Display using chat message style
            with st.chat_message("assistant"):
                if result['type'] == 'dataframe':
                    st.markdown(f"### {result['title']}")
                    st.dataframe(result['data'], hide_index=True, use_container_width=True)
                    
                elif result['type'] == 'text':
                    st.markdown(result['content'])
                    
                elif result['type'] == 'mixed':
                    st.markdown(result['text'])
                    st.dataframe(result['data'], hide_index=True, use_container_width=True)
        else:
            st.info("💬 Ask me anything about the music data!")