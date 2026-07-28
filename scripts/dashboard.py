import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. DATA PROCESSING
# ==============================================================================

# Load the Stata dataset natively
df_raw = pd.read_stata('data/HCMST.dta', convert_categoricals=True)

def find_first_existing(candidates, available_cols):
    for col in candidates:
        if col in available_cols:
            return col
    return None

available = list(df_raw.columns)

column_map = {
    find_first_existing(['w1_ppage', 'w2_ppage', 'w3_ppage', 'pp6_ppage', 'ppage'], available): 'respondent_age',
    find_first_existing(['w1_q9', 'q9', 'partner_age'], available): 'partner_age',
    find_first_existing(['w1_q21b_month', 'q21b_month'], available): 'start_month',
    find_first_existing(['w1_q21c_month', 'q21c_month'], available): 'cohabit_month',
    find_first_existing(['w1_q21d_month', 'q21d_month'], available): 'marriage_month',
    find_first_existing(['w3_breakup_year', 'w2_breakup_year', 'w1_q21e_year', 'w3_Q21E_year'], available): 'breakup_year',
    find_first_existing(['w3_breakup_month', 'w2_breakup_month', 'w1_q21e_month', 'w3_Q21E_month'], available): 'breakup_month',
    find_first_existing(['w3_relationship_duration_yrs', 'w2_relationship_duration', 'W2_RELATIONSHIP_DURATION'], available): 'duration_years',
    find_first_existing(['w1_q32', 'q32'], available): 'how_they_met',
    find_first_existing(['w2_fight', 'w1_q34'], available): 'fight_frequency'
}

valid_mappings = {k: v for k, v in column_map.items() if k is not None}
df = df_raw[list(valid_mappings.keys())].rename(columns=valid_mappings).copy()

# Clean numeric variables & scrub impossible survey codes (-1, 99)
# Clean numeric variables & scrub impossible survey codes (-1, 99)
month_string_map = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}

for col in df.columns:
    if 'month' in col:
        # If the month is a categorical string label from Stata, map it back to 1-12
        if df[col].dtype == 'category' or df[col].dtype == 'object':
            mapped_months = df[col].astype(str).str.lower().str.strip().map(month_string_map)
            # Use the mapped integers, fallback to standard numeric coercion if unmapped
            df[col] = mapped_months.fillna(pd.to_numeric(df[col], errors='coerce'))
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    elif 'year' in col or 'age' in col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'respondent_age' in df.columns:
    df.loc[(df['respondent_age'] < 18) | (df['respondent_age'] > 115), 'respondent_age'] = np.nan
if 'partner_age' in df.columns:
    df.loc[(df['partner_age'] < 18) | (df['partner_age'] > 115), 'partner_age'] = np.nan

if 'respondent_age' in df.columns:
    df.loc[(df['respondent_age'] < 18) | (df['respondent_age'] > 115), 'respondent_age'] = np.nan
if 'partner_age' in df.columns:
    df.loc[(df['partner_age'] < 18) | (df['partner_age'] > 115), 'partner_age'] = np.nan

# Feature Engineering
if 'respondent_age' in df.columns and 'partner_age' in df.columns:
    df['age_gap'] = (df['respondent_age'] - df['partner_age']).abs()

df['has_broken_up'] = df['breakup_year'].notna()
df_breakups = df[df['has_broken_up']].copy()
df_success = df[~df['has_broken_up']].copy()


# ==============================================================================
# 2. VISUALIZATION GENERATION (Pastel Styling)
# ==============================================================================

pastel_palette = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E8BAFF']

transparent_layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#555555', family='Helvetica Neue, sans-serif'),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False),
    margin=dict(l=20, r=20, t=40, b=20)
)

# --- PANEL 1: CUSTOM HTML TIMELINE (Most Common Months) ---
events_map = {
    'start_month': ('Meet', pastel_palette[4]), 
    'cohabit_month': ('Move In', pastel_palette[2]),
    'marriage_month': ('Marry', pastel_palette[3]), 
    'breakup_month': ('Break Up', pastel_palette[0])
}

# Find the peak month for each milestone
month_events = {m: [] for m in range(1, 13)}
for col, (name, color) in events_map.items():
    if col in df.columns:
        counts = df[col].value_counts()
        if not counts.empty:
            peak_month = int(counts.idxmax())
            if 1 <= peak_month <= 12:
                month_events[peak_month].append((name, color))

month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

# Build the custom HTML block
html_timeline = '<h2 class="timeline-title">Peak Relationship Milestones</h2><div class="timeline-container"><div class="timeline-track"></div>'
for m in range(1, 13):
    badges_html = "".join([f'<div class="event-badge" style="background-color: {color};">{name}</div>' for name, color in month_events[m]])
    html_timeline += f'''
    <div class="timeline-month">
        <div class="events-container">{badges_html}</div>
        <div class="month-dot"></div>
        <div class="month-label">{month_names[m]}</div>
    </div>
    '''
html_timeline += '</div>'


# --- PANELS 2 & 3: PLOTLY PROFILES ---
def create_profile_panel(data_subset, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Age vs. Duration", "Fight Frequency"))
    
    fig.add_trace(go.Scatter(
        x=data_subset['respondent_age'], y=data_subset['duration_years'],
        mode='markers', marker=dict(color=pastel_palette[1], size=6, opacity=0.6, line=dict(width=0)),
        name="Couples"
    ), row=1, col=1)
    
    if 'fight_frequency' in data_subset.columns:
        fights = data_subset['fight_frequency'].value_counts().head(5)
        fig.add_trace(go.Bar(
            x=fights.values, y=fights.index, orientation='h',
            marker=dict(color=pastel_palette[5]), name="Fights"
        ), row=1, col=2)
    
    fig.update_layout(**transparent_layout, title=title, showlegend=False)
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_yaxes(title_text="Duration (Years)", row=1, col=1)
    return fig

fig_breakups = create_profile_panel(df_breakups, "Breakup Profiles")
fig_success = create_profile_panel(df_success, "Successful Couple Profiles")


# ==============================================================================
# 3. STATIC HTML EXPORT
# ==============================================================================

# Extract interactive HTML for the bottom plots
html_breakups = fig_breakups.to_html(full_html=False, include_plotlyjs='cdn')
html_success = fig_success.to_html(full_html=False, include_plotlyjs=False)

html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relationship Dynamics Dashboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="dashboard-grid">
        <div class="glass-panel">
            {html_timeline}
        </div>
        <div class="bottom-row">
            <div class="glass-panel">
                {html_breakups}
            </div>
            <div class="glass-panel">
                {html_success}
            </div>
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("Successfully generated index.html with custom timeline.")