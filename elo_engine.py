import streamlit as st
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import math
from datetime import datetime
from scipy import stats
import plotly.figure_factory as ff

# Configure Streamlit page
st.set_page_config(
    page_title="Premier League ELO System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1e3d59;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e5266;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1e3d59;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


@st.cache_resource
def load_and_process_data():
    """Load and process the Premier League data"""
    try:
        # Load and clean the data
        df = pd.read_csv("premier-league-matches.csv")
        df = df[["Date", "Home", "Away", "HomeGoals", "AwayGoals", "FTR"]].copy()
        df.columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Elo parameters
        INITIAL_ELO = 1000
        K = 10
        DECAY_RATE = 0.995
        LOSS_DAMPENING = 0.9
        WIN_SCALING = 0.96
        HOME_ADVANTAGE = 25

        # Dictionary to hold ratings and match history
        elo_ratings = defaultdict(lambda: INITIAL_ELO)
        rating_history = defaultdict(list)
        match_dates = []

        # Track goal statistics for each team
        team_stats = defaultdict(lambda: {
            'goals_scored': [],
            'goals_conceded': [],
            'home_goals_scored': [],
            'home_goals_conceded': [],
            'away_goals_scored': [],
            'away_goals_conceded': [],
            'recent_form': []
        })

        # Time-weighted K factor function
        def time_weighted_k(match_date, latest_date, base_k=40):
            days_diff = (latest_date - match_date).days
            weight = np.exp(-days_diff / 365)
            return base_k * (0.5 + 0.5 * weight)

        latest_date = df["date"].max()

        # Process each match
        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            result = row["result"]
            hg = row["home_goals"]
            ag = row["away_goals"]
            match_date = row["date"]

            # Track goal statistics
            team_stats[home]['goals_scored'].append(hg)
            team_stats[home]['goals_conceded'].append(ag)
            team_stats[home]['home_goals_scored'].append(hg)
            team_stats[home]['home_goals_conceded'].append(ag)

            team_stats[away]['goals_scored'].append(ag)
            team_stats[away]['goals_conceded'].append(hg)
            team_stats[away]['away_goals_scored'].append(ag)
            team_stats[away]['away_goals_conceded'].append(hg)

            # Track recent form
            team_stats[home]['recent_form'].append(hg)
            team_stats[away]['recent_form'].append(ag)

            if len(team_stats[home]['recent_form']) > 10:
                team_stats[home]['recent_form'] = team_stats[home]['recent_form'][-10:]
            if len(team_stats[away]['recent_form']) > 10:
                team_stats[away]['recent_form'] = team_stats[away]['recent_form'][-10:]

            home_rating = elo_ratings[home] + HOME_ADVANTAGE
            away_rating = elo_ratings[away]

            expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
            expected_away = 1 - expected_home

            # Actual score
            if result == "H":
                score_home, score_away = 1.0, 0.0
            elif result == "A":
                score_home, score_away = 0.0, 1.0
            else:
                score_home = score_away = 0.5

            # Goal difference and multiplier
            goal_diff = abs(hg - ag)

            if goal_diff == 0:
                multiplier_home = multiplier_away = 1
            else:
                gap_home = max(0, (away_rating - home_rating) / 400)
                gap_away = max(0, (home_rating - away_rating) / 400)
                multiplier_home = math.log(goal_diff + 1) * (1 + gap_home)
                multiplier_away = math.log(goal_diff + 1) * (1 + gap_away)

            # Time-weighted K factor
            k_factor = time_weighted_k(match_date, latest_date, K)

            base_change_home = k_factor * multiplier_home * (score_home - expected_home)
            base_change_away = k_factor * multiplier_away * (score_away - expected_away)

            # Apply balanced dampening - reduce both wins and losses proportionally
            if base_change_home > 0:  # Home team gaining rating
                base_change_home *= WIN_SCALING
            else:  # Home team losing rating
                base_change_home *= LOSS_DAMPENING

            if base_change_away > 0:  # Away team gaining rating
                base_change_away *= WIN_SCALING
            else:  # Away team losing rating
                base_change_away *= LOSS_DAMPENING

            # Apply the changes
            elo_ratings[home] += base_change_home
            elo_ratings[away] += base_change_away

            # Store rating history
            if idx % 10 == 0:
                match_dates.append(match_date)
                for team in elo_ratings:
                    rating_history[team].append(elo_ratings[team])

        # Apply time decay
        for team in elo_ratings:
            days_since_last = (latest_date - df[df["home_team"] == team]["date"].max()).days
            if pd.isna(days_since_last):
                days_since_last = (latest_date - df[df["away_team"] == team]["date"].max()).days

            if not pd.isna(days_since_last):
                decay_factor = DECAY_RATE ** (days_since_last / 30)
                elo_ratings[team] *= decay_factor

        # Create final DataFrame
        elo_df = pd.DataFrame(elo_ratings.items(), columns=["Team", "Final_Elo"])
        elo_df = elo_df.sort_values(by="Final_Elo", ascending=False).reset_index(drop=True)
        elo_df['Rank'] = range(1, len(elo_df) + 1)

        return elo_df, elo_ratings, team_stats, rating_history, match_dates

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None


class ScorePredictor:
    def __init__(self, elo_ratings, team_stats, league_avg_goals=2.7):
        self.elo_ratings = elo_ratings
        self.team_stats = team_stats
        self.league_avg_goals = league_avg_goals
        self.home_advantage = 0.15

    def get_team_attack_strength(self, team, is_home=True):
        if is_home:
            goals = self.team_stats[team]['home_goals_scored']
        else:
            goals = self.team_stats[team]['away_goals_scored']

        if not goals:
            return 1.0

        avg_goals = np.mean(goals)
        return avg_goals / (self.league_avg_goals / 2)

    def get_team_defense_strength(self, team, is_home=True):
        if is_home:
            goals = self.team_stats[team]['home_goals_conceded']
        else:
            goals = self.team_stats[team]['away_goals_conceded']

        if not goals:
            return 1.0

        avg_conceded = np.mean(goals)
        return (self.league_avg_goals / 2) / avg_conceded

    def elo_to_goal_expectation(self, elo_diff):
        return 1 + (elo_diff / 100) * 0.3

    def predict_goals(self, home_team, away_team):
        home_elo = self.elo_ratings[home_team]
        away_elo = self.elo_ratings[away_team]

        base_expectation = self.league_avg_goals / 2

        elo_diff_home = home_elo - away_elo
        elo_diff_away = away_elo - home_elo

        elo_mult_home = self.elo_to_goal_expectation(elo_diff_home)
        elo_mult_away = self.elo_to_goal_expectation(elo_diff_away)

        home_attack = self.get_team_attack_strength(home_team, True)
        home_defense = self.get_team_defense_strength(home_team, True)
        away_attack = self.get_team_attack_strength(away_team, False)
        away_defense = self.get_team_defense_strength(away_team, False)

        home_expected = (base_expectation + self.home_advantage) * elo_mult_home * home_attack / away_defense
        away_expected = base_expectation * elo_mult_away * away_attack / home_defense

        home_expected = max(0.1, min(6.0, home_expected))
        away_expected = max(0.1, min(6.0, away_expected))

        return home_expected, away_expected

    def predict_score_probabilities(self, home_team, away_team, max_goals=6):
        home_lambda, away_lambda = self.predict_goals(home_team, away_team)

        probabilities = {}
        total_prob = 0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (stats.poisson.pmf(home_goals, home_lambda) *
                        stats.poisson.pmf(away_goals, away_lambda))
                probabilities[(home_goals, away_goals)] = prob
                total_prob += prob

        for score in probabilities:
            probabilities[score] /= total_prob

        return probabilities, home_lambda, away_lambda

    def predict_match_outcome(self, home_team, away_team):
        probs, home_lambda, away_lambda = self.predict_score_probabilities(home_team, away_team)

        home_win_prob = sum(prob for (hg, ag), prob in probs.items() if hg > ag)
        draw_prob = sum(prob for (hg, ag), prob in probs.items() if hg == ag)
        away_win_prob = sum(prob for (hg, ag), prob in probs.items() if hg < ag)

        most_likely_score = max(probs, key=probs.get)

        return {
            'home_win_prob': home_win_prob,
            'draw_prob': draw_prob,
            'away_win_prob': away_win_prob,
            'most_likely_score': most_likely_score,
            'score_probabilities': probs,
            'expected_goals': (home_lambda, away_lambda)
        }


def get_rating_tier(elo):
    if elo >= 1200:
        return "Elite"
    elif elo >= 1100:
        return "Strong"
    elif elo >= 900:
        return "Average"
    elif elo >= 700:
        return "Weak"
    else:
        return "Very Weak"


def main():
    st.markdown('<h1 class="main-header">⚽ Premier League Elo Prediction System</h1>', unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading and processing Premier League data..."):
        elo_df, elo_ratings, team_stats, rating_history, match_dates = load_and_process_data()

    if elo_df is None:
        st.error("Failed to load data. Please check your data file.")
        return

    # Initialize predictor
    predictor = ScorePredictor(elo_ratings, team_stats)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Team Rankings", "🔮 Match Predictor", "📈 Rating Evolution", "📋 Team Statistics", "🎯 Batch Predictions"]
    )

    if page == "📊 Team Rankings":
        show_team_rankings(elo_df)
    elif page == "🔮 Match Predictor":
        show_match_predictor(elo_df, predictor)
    elif page == "📈 Rating Evolution":
        show_rating_evolution(elo_df, rating_history, match_dates)
    elif page == "📋 Team Statistics":
        show_team_statistics(elo_df, team_stats)
    elif page == "🎯 Batch Predictions":
        show_batch_predictions(elo_df, predictor)


def show_team_rankings(elo_df):
    st.markdown('<h2 class="section-header">Team Rankings</h2>', unsafe_allow_html=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Teams", len(elo_df))
    with col2:
        st.metric("Highest Rating", f"{elo_df['Final_Elo'].max():.1f}")
    with col3:
        st.metric("Average Rating", f"{elo_df['Final_Elo'].mean():.1f}")
    with col4:
        st.metric("Lowest Rating", f"{elo_df['Final_Elo'].min():.1f}")

    # Tier distribution
    elo_df['Tier'] = elo_df['Final_Elo'].apply(get_rating_tier)
    tier_counts = elo_df['Tier'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Tier Distribution")
        fig = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            title="Teams by Rating Tier"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 Teams")
        top_teams = elo_df.head(15)
        fig = px.bar(
            top_teams,
            x='Team',
            y='Final_Elo',
            title="Top 15 Teams by Elo Rating",
            color='Final_Elo',
            color_continuous_scale='RdYlGn'
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Full rankings table
    st.subheader("Complete Rankings")

    # Add search functionality
    search_term = st.text_input("Search for a team:")

    display_df = elo_df.copy()
    if search_term:
        display_df = display_df[display_df['Team'].str.contains(search_term, case=False)]

    # Color coding based on tiers
    def highlight_tier(row):
        tier = get_rating_tier(row['Final_Elo'])
        if tier == "Elite":
            return ['background-color: #90EE90'] * len(row)
        elif tier == "Strong":
            return ['background-color: #FFD700'] * len(row)
        elif tier == "Average":
            return ['background-color: #FFA500'] * len(row)
        elif tier == "Weak":
            return ['background-color: #FFB6C1'] * len(row)
        else:
            return ['background-color: #FF6B6B'] * len(row)

    styled_df = display_df.style.apply(highlight_tier, axis=1)
    st.dataframe(styled_df, use_container_width=True)


def show_match_predictor(elo_df, predictor):
    st.markdown('<h2 class="section-header">Match Predictor</h2>', unsafe_allow_html=True)

    teams = sorted(elo_df['Team'].tolist())

    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Select Home Team:", teams, key="home_team")

    with col2:
        away_team = st.selectbox("Select Away Team:", teams, key="away_team")

    if home_team == away_team:
        st.warning("Please select different teams for home and away.")
        return

    # Store prediction on button click
    if st.button("Predict Match", type="primary"):
        with st.spinner("Calculating prediction..."):
            prediction = predictor.predict_match_outcome(home_team, away_team)
            st.session_state.prediction = prediction
            st.session_state.prediction_teams = (home_team, away_team)

    # Display prediction if available
    if 'prediction' in st.session_state and 'prediction_teams' in st.session_state:
        prediction = st.session_state.prediction
        home_team, away_team = st.session_state.prediction_teams

        # Team information
        home_elo = predictor.elo_ratings[home_team]
        away_elo = predictor.elo_ratings[away_team]
        elo_diff = home_elo - away_elo

        st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"### 🏟️ {home_team} vs {away_team}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Home Team Elo", f"{home_elo:.0f}")
        with col2:
            st.metric("Away Team Elo", f"{away_elo:.0f}")
        with col3:
            st.metric("Elo Difference", f"{elo_diff:+.0f}")

        # Expected goals
        exp_home, exp_away = prediction['expected_goals']
        st.markdown(f"**Expected Goals:** {home_team} {exp_home:.2f} - {exp_away:.2f} {away_team}")

        # Most likely score
        most_likely = prediction['most_likely_score']
        st.markdown(f"**Most Likely Score:** {most_likely[0]}-{most_likely[1]}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Outcome probabilities
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(f"{home_team} Win", f"{prediction['home_win_prob']:.1%}")
        with col2:
            st.metric("Draw", f"{prediction['draw_prob']:.1%}")
        with col3:
            st.metric(f"{away_team} Win", f"{prediction['away_win_prob']:.1%}")

        # Probability visualization
        outcomes = ['Home Win', 'Draw', 'Away Win']
        probabilities = [prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob']]

        fig = px.bar(
            x=outcomes,
            y=probabilities,
            title="Match Outcome Probabilities",
            color=probabilities,
            color_continuous_scale='RdYlGn'
        )
        fig.update_yaxes(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 most likely scores
        st.subheader("Top 10 Most Likely Scores")
        top_scores = sorted(prediction['score_probabilities'].items(), key=lambda x: x[1], reverse=True)[:10]

        scores_df = pd.DataFrame([
            {'Score': f"{hg}-{ag}", 'Probability': f"{prob:.1%}"}
            for (hg, ag), prob in top_scores
        ])

        st.dataframe(scores_df, use_container_width=True, hide_index=True)

        # Score probability heatmap
        if st.checkbox("Show Score Probability Heatmap"):
            show_score_heatmap(prediction['score_probabilities'], home_team, away_team)


def show_score_heatmap(probabilities, home_team, away_team):
    # Create matrix for heatmap
    matrix = np.zeros((7, 7))
    for (hg, ag), prob in probabilities.items():
        if hg <= 6 and ag <= 6:
            matrix[ag, hg] = prob

    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z=matrix,
        x=list(range(7)),
        y=list(range(7)),
        colorscale='YlOrRd',
        annotation_text=[[f"{val:.3f}" for val in row] for row in matrix],
        showscale=True
    )

    fig.update_layout(
        title=f'Score Probability Matrix: {home_team} vs {away_team}',
        xaxis_title=f'{home_team} Goals',
        yaxis_title=f'{away_team} Goals',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)


def show_rating_evolution(elo_df, rating_history, match_dates):
    st.markdown('<h2 class="section-header">Rating Evolution Over Time</h2>', unsafe_allow_html=True)

    # Team selection
    teams = elo_df['Team'].tolist()
    selected_teams = st.multiselect(
        "Select teams to display:",
        teams,
        default=teams[:5]  # Default to top 5 teams
    )

    if not selected_teams:
        st.warning("Please select at least one team.")
        return

    # Create evolution plot
    fig = go.Figure()

    colors = px.colors.qualitative.Set1

    for i, team in enumerate(selected_teams):
        if team in rating_history and len(rating_history[team]) > 0:
            team_ratings = rating_history[team][:len(match_dates)]
            if len(team_ratings) > 0:
                fig.add_trace(go.Scatter(
                    x=match_dates[:len(team_ratings)],
                    y=team_ratings,
                    mode='lines+markers',
                    name=team,
                    line=dict(width=2, color=colors[i % len(colors)]),
                    marker=dict(size=4)
                ))

    fig.update_layout(
        title="Elo Rating Evolution Over Time",
        xaxis_title="Date",
        yaxis_title="Elo Rating",
        hovermode='x unified',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Rating distribution
    st.subheader("Current Rating Distribution")
    fig = px.histogram(
        elo_df,
        x='Final_Elo',
        nbins=20,
        title="Distribution of Current Elo Ratings",
        marginal="box"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_team_statistics(elo_df, team_stats):
    st.markdown('<h2 class="section-header">Team Statistics</h2>', unsafe_allow_html=True)

    team = st.selectbox("Select a team:", elo_df['Team'].tolist())

    if team in team_stats:
        stats = team_stats[team]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Goal Statistics")
            if stats['goals_scored']:
                st.metric("Average Goals Scored", f"{np.mean(stats['goals_scored']):.2f}")
                st.metric("Average Goals Conceded", f"{np.mean(stats['goals_conceded']):.2f}")
                st.metric("Goal Difference", f"{np.mean(stats['goals_scored']) - np.mean(stats['goals_conceded']):.2f}")

        with col2:
            st.subheader("Home vs Away Performance")
            if stats['home_goals_scored']:
                st.metric("Home Goals/Game", f"{np.mean(stats['home_goals_scored']):.2f}")
                st.metric("Away Goals/Game", f"{np.mean(stats['away_goals_scored']):.2f}")
            if stats['home_goals_conceded']:
                st.metric("Home Goals Conceded/Game", f"{np.mean(stats['home_goals_conceded']):.2f}")
                st.metric("Away Goals Conceded/Game", f"{np.mean(stats['away_goals_conceded']):.2f}")

        # Goal distribution charts
        if stats['goals_scored']:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    x=stats['goals_scored'],
                    title=f"{team} - Goals Scored Distribution",
                    labels={'x': 'Goals Scored', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    x=stats['goals_conceded'],
                    title=f"{team} - Goals Conceded Distribution",
                    labels={'x': 'Goals Conceded', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)


def show_batch_predictions(elo_df, predictor):
    st.markdown('<h2 class="section-header">Batch Predictions</h2>', unsafe_allow_html=True)

    st.write("Enter multiple matches to predict (one per line in format: Home Team vs Away Team)")

    # Text area for batch input
    matches_input = st.text_area(
        "Enter matches:",
        placeholder="Arsenal vs Chelsea\nLiverpool vs Manchester City\nTottenham vs Manchester United",
        height=150
    )

    teams = elo_df['Team'].tolist()

    if st.button("Predict All Matches", type="primary"):
        if not matches_input.strip():
            st.warning("Please enter at least one match.")
            return

        lines = matches_input.strip().split('\n')
        results = []

        for line in lines:
            if ' vs ' in line:
                home_input, away_input = line.split(' vs ', 1)
                home_input = home_input.strip()
                away_input = away_input.strip()

                # Find matching teams
                home_team = None
                away_team = None

                for team in teams:
                    if home_input.lower() in team.lower() or team.lower() in home_input.lower():
                        home_team = team
                        break

                for team in teams:
                    if away_input.lower() in team.lower() or team.lower() in away_input.lower():
                        away_team = team
                        break

                if home_team and away_team and home_team != away_team:
                    prediction = predictor.predict_match_outcome(home_team, away_team)
                    results.append({
                        'Match': f"{home_team} vs {away_team}",
                        'Expected Goals': f"{prediction['expected_goals'][0]:.2f} - {prediction['expected_goals'][1]:.2f}",
                        'Most Likely Score': f"{prediction['most_likely_score'][0]}-{prediction['most_likely_score'][1]}",
                        'Home Win %': f"{prediction['home_win_prob']:.1%}",
                        'Draw %': f"{prediction['draw_prob']:.1%}",
                        'Away Win %': f"{prediction['away_win_prob']:.1%}"
                    })
                else:
                    st.warning(f"Could not find teams for: {line}")

        if results:
            st.subheader("Prediction Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name="match_predictions.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()