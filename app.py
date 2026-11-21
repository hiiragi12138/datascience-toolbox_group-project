from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


# 1. Load the cleaned data

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "instagram_analytics_cleaned.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Could not find instagram_analytics_cleaned.csv. "
        "Make sure it is in the same folder as app.py."
    )

df = pd.read_csv(DATA_PATH, parse_dates=["post_datetime", "post_date"])

# Support both True/False and 1/0.
df["has_call_to_action"] = (
    df["has_call_to_action"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1"])
)

df["CTA"] = np.where(df["has_call_to_action"], "With CTA", "Without CTA")


# 2. Summarize by post format, content topic, and CTA status

controlled = (
    df.groupby(
        ["media_type", "content_category", "has_call_to_action"],
        observed=True,
    )
    .agg(
        posts=("post_id", "count"),
        avg_reach=("reach", "mean"),
        avg_engagement_rate=("engagement_rate", "mean"),
        avg_followers_gained=("followers_gained", "mean"),
    )
    .reset_index()
)

content_keys = ["media_type", "content_category"]
content_yes = controlled.loc[controlled["has_call_to_action"]].drop(
    columns="has_call_to_action"
)
content_no = controlled.loc[~controlled["has_call_to_action"]].drop(
    columns="has_call_to_action"
)

content_diff = content_yes.merge(
    content_no,
    on=content_keys,
    suffixes=("_yes", "_no"),
    how="inner",
)

content_diff["reach_diff"] = (
    content_diff["avg_reach_yes"] - content_diff["avg_reach_no"]
)
content_diff["engagement_diff_pp"] = (
    content_diff["avg_engagement_rate_yes"]
    - content_diff["avg_engagement_rate_no"]
) * 100
content_diff["followers_diff"] = (
    content_diff["avg_followers_gained_yes"]
    - content_diff["avg_followers_gained_no"]
)


# 3. Compare CTA performance within each account

account_cta = (
    df.groupby(
        ["account_id", "media_type", "content_category", "has_call_to_action"],
        observed=True,
    )
    .agg(
        posts=("post_id", "count"),
        avg_reach=("reach", "mean"),
        avg_engagement_rate=("engagement_rate", "mean"),
        avg_followers_gained=("followers_gained", "mean"),
    )
    .reset_index()
)

account_keys = ["account_id", "media_type", "content_category"]
account_yes = account_cta.loc[account_cta["has_call_to_action"]].drop(
    columns="has_call_to_action"
)
account_no = account_cta.loc[~account_cta["has_call_to_action"]].drop(
    columns="has_call_to_action"
)

paired = account_yes.merge(
    account_no,
    on=account_keys,
    suffixes=("_yes", "_no"),
    how="inner",
)

paired["reach_diff"] = paired["avg_reach_yes"] - paired["avg_reach_no"]
paired["engagement_diff_pp"] = (
    paired["avg_engagement_rate_yes"] - paired["avg_engagement_rate_no"]
) * 100
paired["followers_diff"] = (
    paired["avg_followers_gained_yes"] - paired["avg_followers_gained_no"]
)


# 4. Calculate consistency across accounts

account_consistency = (
    paired.groupby(["media_type", "content_category"], observed=True)
    .agg(
        paired_accounts=("account_id", "nunique"),
        reach_positive_accounts=(
            "reach_diff",
            lambda values: int((values > 0).sum()),
        ),
        engagement_positive_accounts=(
            "engagement_diff_pp",
            lambda values: int((values > 0).sum()),
        ),
        followers_positive_accounts=(
            "followers_diff",
            lambda values: int((values > 0).sum()),
        ),
        median_reach_diff=("reach_diff", "median"),
        median_engagement_diff_pp=("engagement_diff_pp", "median"),
        median_followers_diff=("followers_diff", "median"),
    )
    .reset_index()
)


# 5. Initialize Dash

app = Dash(__name__)

# Render uses this server.
server = app.server

app.title = "Q2 CTA Strategy Dashboard"


# 6. Filter options

MEDIA_OPTIONS = [{"label": "All post formats", "value": "ALL"}] + [
    {"label": value.title(), "value": value}
    for value in sorted(df["media_type"].unique())
]

CATEGORY_OPTIONS = [{"label": "All content topics", "value": "ALL"}] + [
    {"label": value, "value": value}
    for value in sorted(df["content_category"].unique())
]


# 7. Page styles

PAGE_STYLE = {
    "backgroundColor": "#F4F6F8",
    "minHeight": "100vh",
    "padding": "28px",
    "fontFamily": "Arial, sans-serif",
}

CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "18px",
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
}

SECTION_STYLE = {
    **CARD_STYLE,
    "marginTop": "20px",
}


# 8. Dashboard layout

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.H1("Q2 CTA Strategy Dashboard"),
        html.P(
            "Compare historical performance for posts with and without a CTA "
            "across post formats and content topics."
        ),

        # Filters
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
                "gap": "16px",
            },
            children=[
                html.Div([
                    html.Label("Post Format"),
                    dcc.Dropdown(
                        id="media-filter",
                        options=MEDIA_OPTIONS,
                        value="ALL",
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Content Topic"),
                    dcc.Dropdown(
                        id="category-filter",
                        options=CATEGORY_OPTIONS,
                        value="ALL",
                        clearable=False,
                    ),
                ]),
            ],
        ),

        # KPI cards
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
                "gap": "16px",
                "marginTop": "20px",
            },
            children=[
                html.Div(id="reach-card", style=CARD_STYLE),
                html.Div(id="engagement-card", style=CARD_STYLE),
                html.Div(id="followers-card", style=CARD_STYLE),
            ],
        ),

        # Average performance with and without a CTA
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("Average Performance With and Without a CTA"),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": (
                            "repeat(auto-fit, minmax(300px, 1fr))"
                        ),
                        "gap": "10px",
                    },
                    children=[
                        dcc.Graph(id="reach-chart"),
                        dcc.Graph(id="engagement-chart"),
                        dcc.Graph(id="followers-chart"),
                    ],
                ),
            ],
        ),

        # Heatmap of differences by post format and topic
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("CTA Differences by Post Format and Topic"),
                html.P(
                    "Each difference equals the value for posts with a CTA "
                    "minus the value for posts without a CTA."
                ),
                dcc.RadioItems(
                    id="heatmap-metric",
                    options=[
                        {"label": "Average Reach", "value": "reach_diff"},
                        {
                            "label": "Average Engagement Rate",
                            "value": "engagement_diff_pp",
                        },
                        {
                            "label": "Average Followers Gained",
                            "value": "followers_diff",
                        },
                    ],
                    value="reach_diff",
                    inline=True,
                ),
                dcc.Graph(id="difference-heatmap"),
            ],
        ),

        # Consistency across accounts
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("CTA Performance Across Accounts"),
                html.P(
                    "The dashed black line marks 50%. Bars above it indicate "
                    "that posts with a CTA performed better in more than half "
                    "of the accounts."
                ),
                dcc.Graph(id="consistency-chart"),
            ],
        ),

        # Dynamic summary
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("Results for the Current Filters"),
                html.Div(id="dynamic-insight"),
            ],
        ),
    ],
)


# 9. Helper functions

def make_card(title, yes_value, no_value, unit="", decimals=2):
    difference = yes_value - no_value

    return [
        html.H3(title, style={"marginTop": "0"}),
        html.P(f"With CTA: {yes_value:,.{decimals}f}{unit}"),
        html.P(f"Without CTA: {no_value:,.{decimals}f}{unit}"),
        html.Strong(f"Difference: {difference:+,.{decimals}f}{unit}"),
    ]


def make_bar(data, value_column, title, y_title):
    figure = px.bar(
        data,
        x="CTA",
        y=value_column,
        color="CTA",
        text=value_column,
        title=title,
        category_orders={"CTA": ["Without CTA", "With CTA"]},
        color_discrete_map={
            "Without CTA": "#9E9E9E",
            "With CTA": "#4C78A8",
        },
    )

    figure.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    figure.update_layout(xaxis_title="", yaxis_title=y_title, showlegend=False)

    return figure


# 10. Dashboard callbacks

@app.callback(
    Output("reach-card", "children"),
    Output("engagement-card", "children"),
    Output("followers-card", "children"),
    Output("reach-chart", "figure"),
    Output("engagement-chart", "figure"),
    Output("followers-chart", "figure"),
    Output("difference-heatmap", "figure"),
    Output("consistency-chart", "figure"),
    Output("dynamic-insight", "children"),
    Input("media-filter", "value"),
    Input("category-filter", "value"),
    Input("heatmap-metric", "value"),
)
def update_dashboard(selected_media, selected_category, selected_metric):
    # Filter the source data.
    filtered = df.copy()

    if selected_media != "ALL":
        filtered = filtered.loc[filtered["media_type"] == selected_media]

    if selected_category != "ALL":
        filtered = filtered.loc[filtered["content_category"] == selected_category]

    # Calculate averages for posts with and without a CTA.
    summary = (
        filtered.groupby("CTA", observed=True)
        .agg(
            posts=("post_id", "count"),
            avg_reach=("reach", "mean"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_followers_gained=("followers_gained", "mean"),
        )
        .reset_index()
    )

    summary["avg_engagement_rate_pct"] = summary["avg_engagement_rate"] * 100
    lookup = summary.set_index("CTA")

    yes_reach = float(lookup.loc["With CTA", "avg_reach"])
    no_reach = float(lookup.loc["Without CTA", "avg_reach"])
    yes_engagement = float(lookup.loc["With CTA", "avg_engagement_rate_pct"])
    no_engagement = float(lookup.loc["Without CTA", "avg_engagement_rate_pct"])
    yes_followers = float(lookup.loc["With CTA", "avg_followers_gained"])
    no_followers = float(lookup.loc["Without CTA", "avg_followers_gained"])

    # Build the KPI cards.
    reach_card = make_card("Average Reach", yes_reach, no_reach)
    engagement_card = make_card(
        "Average Engagement Rate",
        yes_engagement,
        no_engagement,
        unit="%",
        decimals=3,
    )
    followers_card = make_card(
        "Average Followers Gained",
        yes_followers,
        no_followers,
    )

    # Build the three bar charts.
    reach_figure = make_bar(
        summary, "avg_reach", "Average Reach", "Average Reach"
    )
    engagement_figure = make_bar(
        summary,
        "avg_engagement_rate_pct",
        "Average Engagement Rate",
        "Average Engagement Rate (%)",
    )
    followers_figure = make_bar(
        summary,
        "avg_followers_gained",
        "Average Followers Gained",
        "Average Followers Gained",
    )

    # Filter differences by post format and topic.
    selected_diff = content_diff.copy()

    if selected_media != "ALL":
        selected_diff = selected_diff.loc[
            selected_diff["media_type"] == selected_media
        ]

    if selected_category != "ALL":
        selected_diff = selected_diff.loc[
            selected_diff["content_category"] == selected_category
        ]

    heatmap_settings = {
        "reach_diff": (
            "Difference in Average Reach With a CTA",
            "Reach Difference",
            ".1f",
        ),
        "engagement_diff_pp": (
            "Difference in Average Engagement Rate With a CTA",
            "Difference (Percentage Points)",
            ".3f",
        ),
        "followers_diff": (
            "Difference in Average Followers Gained With a CTA",
            "Difference in Followers Gained",
            ".1f",
        ),
    }

    heatmap_title, color_title, number_format = heatmap_settings[selected_metric]

    heatmap_data = selected_diff.pivot(
        index="content_category",
        columns="media_type",
        values=selected_metric,
    )

    heatmap_figure = px.imshow(
        heatmap_data,
        text_auto=number_format,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=heatmap_title,
        labels={"x": "Post Format", "y": "Content Topic", "color": color_title},
    )
    heatmap_figure.update_xaxes(side="top")

    # Filter account-level consistency results.
    selected_consistency = account_consistency.copy()

    if selected_media != "ALL":
        selected_consistency = selected_consistency.loc[
            selected_consistency["media_type"] == selected_media
        ]

    if selected_category != "ALL":
        selected_consistency = selected_consistency.loc[
            selected_consistency["content_category"] == selected_category
        ]

    consistency_long = selected_consistency.melt(
        id_vars=["media_type", "content_category", "paired_accounts"],
        value_vars=[
            "reach_positive_accounts",
            "engagement_positive_accounts",
            "followers_positive_accounts",
        ],
        var_name="metric",
        value_name="positive_accounts",
    )

    consistency_long["positive_percentage"] = (
        consistency_long["positive_accounts"]
        / consistency_long["paired_accounts"]
        * 100
    )

    consistency_long["metric"] = consistency_long["metric"].map({
        "reach_positive_accounts": "Reach",
        "engagement_positive_accounts": "Engagement Rate",
        "followers_positive_accounts": "Followers Gained",
    })

    consistency_figure = px.bar(
        consistency_long,
        x="content_category",
        y="positive_percentage",
        color="metric",
        barmode="group",
        facet_col="media_type",
        title="Share of Accounts With Better Performance on Posts With a CTA",
        labels={
            "content_category": "Content Topic",
            "positive_percentage": "Share of Accounts (%)",
            "metric": "Metric",
        },
    )

    consistency_figure.add_hline(y=50, line_dash="dash", line_color="black")
    consistency_figure.update_yaxes(range=[0, 100])
    consistency_figure.update_xaxes(tickangle=45)
    consistency_figure.for_each_annotation(
        lambda annotation: annotation.update(
            text=annotation.text.split("=")[-1].title()
        )
    )

    # Generate the summary for the current filters.
    media_text = (
        "All post formats" if selected_media == "ALL" else selected_media.title()
    )
    category_text = (
        "All content topics" if selected_category == "ALL" else selected_category
    )

    insight = html.P([
        f"Current filters: {media_text}, {category_text}.",
        html.Br(),
        "For posts with a CTA compared with posts without a CTA, "
        f"the difference in average reach was {yes_reach - no_reach:+,.2f}. ",
        "The difference in average engagement rate was "
        f"{yes_engagement - no_engagement:+.3f} percentage points. ",
        "The difference in average followers gained was "
        f"{yes_followers - no_followers:+,.2f}.",
        html.Br(),
        "These differences reflect associations in historical data. "
        "The data do not establish whether the CTA caused the differences.",
    ])

    return (
        reach_card,
        engagement_card,
        followers_card,
        reach_figure,
        engagement_figure,
        followers_figure,
        heatmap_figure,
        consistency_figure,
        insight,
    )


# 11. Run locally

if __name__ == "__main__":
    app.run(debug=True)
