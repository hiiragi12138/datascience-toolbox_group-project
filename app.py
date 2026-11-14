from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


# ============================================================
# 1. 讀取cleaned資料
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "instagram_analytics_cleaned.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "找不到 instagram_analytics_cleaned.csv，"
        "請確認它和 app.py 放在同一個資料夾。"
    )

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["post_datetime", "post_date"]
)

# 同時支援True/False和1/0
df["has_call_to_action"] = (
    df["has_call_to_action"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1"])
)

df["CTA"] = np.where(
    df["has_call_to_action"],
    "有 CTA",
    "沒有 CTA"
)


# ============================================================
# 2. 建立形式 × 主題 × CTA摘要
# ============================================================

controlled = (
    df.groupby(
        [
            "media_type",
            "content_category",
            "has_call_to_action"
        ],
        observed=True
    )
    .agg(
        posts=("post_id", "count"),
        avg_reach=("reach", "mean"),
        avg_engagement_rate=("engagement_rate", "mean"),
        avg_followers_gained=("followers_gained", "mean")
    )
    .reset_index()
)

content_keys = [
    "media_type",
    "content_category"
]

content_yes = (
    controlled.loc[
        controlled["has_call_to_action"]
    ]
    .drop(columns="has_call_to_action")
)

content_no = (
    controlled.loc[
        ~controlled["has_call_to_action"]
    ]
    .drop(columns="has_call_to_action")
)

content_diff = content_yes.merge(
    content_no,
    on=content_keys,
    suffixes=("_yes", "_no"),
    how="inner"
)

content_diff["reach_diff"] = (
    content_diff["avg_reach_yes"]
    - content_diff["avg_reach_no"]
)

content_diff["engagement_diff_pp"] = (
    content_diff["avg_engagement_rate_yes"]
    - content_diff["avg_engagement_rate_no"]
) * 100

content_diff["followers_diff"] = (
    content_diff["avg_followers_gained_yes"]
    - content_diff["avg_followers_gained_no"]
)


# ============================================================
# 3. 建立同一帳號內的CTA比較
# ============================================================

account_cta = (
    df.groupby(
        [
            "account_id",
            "media_type",
            "content_category",
            "has_call_to_action"
        ],
        observed=True
    )
    .agg(
        posts=("post_id", "count"),
        avg_reach=("reach", "mean"),
        avg_engagement_rate=("engagement_rate", "mean"),
        avg_followers_gained=("followers_gained", "mean")
    )
    .reset_index()
)

account_keys = [
    "account_id",
    "media_type",
    "content_category"
]

account_yes = (
    account_cta.loc[
        account_cta["has_call_to_action"]
    ]
    .drop(columns="has_call_to_action")
)

account_no = (
    account_cta.loc[
        ~account_cta["has_call_to_action"]
    ]
    .drop(columns="has_call_to_action")
)

paired = account_yes.merge(
    account_no,
    on=account_keys,
    suffixes=("_yes", "_no"),
    how="inner"
)

paired["reach_diff"] = (
    paired["avg_reach_yes"]
    - paired["avg_reach_no"]
)

paired["engagement_diff_pp"] = (
    paired["avg_engagement_rate_yes"]
    - paired["avg_engagement_rate_no"]
) * 100

paired["followers_diff"] = (
    paired["avg_followers_gained_yes"]
    - paired["avg_followers_gained_no"]
)


# ============================================================
# 4. 計算不同帳號的一致性
# ============================================================

account_consistency = (
    paired.groupby(
        [
            "media_type",
            "content_category"
        ],
        observed=True
    )
    .agg(
        paired_accounts=(
            "account_id",
            "nunique"
        ),

        reach_positive_accounts=(
            "reach_diff",
            lambda values: int((values > 0).sum())
        ),

        engagement_positive_accounts=(
            "engagement_diff_pp",
            lambda values: int((values > 0).sum())
        ),

        followers_positive_accounts=(
            "followers_diff",
            lambda values: int((values > 0).sum())
        ),

        median_reach_diff=(
            "reach_diff",
            "median"
        ),

        median_engagement_diff_pp=(
            "engagement_diff_pp",
            "median"
        ),

        median_followers_diff=(
            "followers_diff",
            "median"
        )
    )
    .reset_index()
)


# ============================================================
# 5. 建立Dash
# ============================================================

app = Dash(__name__)

# Render使用這個server
server = app.server

app.title = "Q2 CTA Strategy Dashboard"


# ============================================================
# 6. 篩選器選項
# ============================================================

MEDIA_OPTIONS = [
    {
        "label": "全部形式",
        "value": "ALL"
    }
] + [
    {
        "label": value.title(),
        "value": value
    }
    for value in sorted(
        df["media_type"].unique()
    )
]

CATEGORY_OPTIONS = [
    {
        "label": "全部主題",
        "value": "ALL"
    }
] + [
    {
        "label": value,
        "value": value
    }
    for value in sorted(
        df["content_category"].unique()
    )
]


# ============================================================
# 7. 頁面樣式
# ============================================================

PAGE_STYLE = {
    "backgroundColor": "#F4F6F8",
    "minHeight": "100vh",
    "padding": "28px",
    "fontFamily": "Arial, sans-serif"
}

CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "18px",
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
}

SECTION_STYLE = {
    **CARD_STYLE,
    "marginTop": "20px"
}


# ============================================================
# 8. Dashboard頁面
# ============================================================

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[
        html.H1(
            "Q2 CTA Strategy Dashboard"
        ),

        html.P(
            "比較不同帖子形式與內容主題中，"
            "有CTA與沒有CTA的歷史表現。"
        ),

        # 篩選器
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(240px, 1fr))"
                ),
                "gap": "16px"
            },
            children=[
                html.Div([
                    html.Label("帖子形式"),

                    dcc.Dropdown(
                        id="media-filter",
                        options=MEDIA_OPTIONS,
                        value="ALL",
                        clearable=False
                    )
                ]),

                html.Div([
                    html.Label("內容主題"),

                    dcc.Dropdown(
                        id="category-filter",
                        options=CATEGORY_OPTIONS,
                        value="ALL",
                        clearable=False
                    )
                ])
            ]
        ),

        # KPI卡片
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": (
                    "repeat(auto-fit, minmax(240px, 1fr))"
                ),
                "gap": "16px",
                "marginTop": "20px"
            },
            children=[
                html.Div(
                    id="reach-card",
                    style=CARD_STYLE
                ),

                html.Div(
                    id="engagement-card",
                    style=CARD_STYLE
                ),

                html.Div(
                    id="followers-card",
                    style=CARD_STYLE
                )
            ]
        ),

        # 有無CTA的平均表現
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2(
                    "有無CTA的平均表現"
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": (
                            "repeat(auto-fit, minmax(300px, 1fr))"
                        ),
                        "gap": "10px"
                    },
                    children=[
                        dcc.Graph(
                            id="reach-chart"
                        ),

                        dcc.Graph(
                            id="engagement-chart"
                        ),

                        dcc.Graph(
                            id="followers-chart"
                        )
                    ]
                )
            ]
        ),

        # 情境差異熱圖
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2(
                    "不同內容情境的CTA差異"
                ),

                html.P(
                    "差異以「有CTA－沒有CTA」計算。"
                ),

                dcc.RadioItems(
                    id="heatmap-metric",
                    options=[
                        {
                            "label": "平均觸及",
                            "value": "reach_diff"
                        },
                        {
                            "label": "平均互動率",
                            "value": "engagement_diff_pp"
                        },
                        {
                            "label": "平均新增追蹤者",
                            "value": "followers_diff"
                        }
                    ],
                    value="reach_diff",
                    inline=True
                ),

                dcc.Graph(
                    id="difference-heatmap"
                )
            ]
        ),

        # 跨帳號一致性
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2(
                    "不同帳號中的CTA表現一致性"
                ),

                html.P(
                    "黑色虛線代表50%；高於虛線表示"
                    "超過一半帳號的有CTA貼文表現較高。"
                ),

                dcc.Graph(
                    id="consistency-chart"
                )
            ]
        ),

        # 動態結論
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2(
                    "目前篩選條件的結果"
                ),

                html.Div(
                    id="dynamic-insight"
                )
            ]
        )
    ]
)


# ============================================================
# 9. 輔助函數
# ============================================================

def make_card(
    title,
    yes_value,
    no_value,
    unit="",
    decimals=2
):
    difference = yes_value - no_value

    return [
        html.H3(
            title,
            style={"marginTop": "0"}
        ),

        html.P(
            f"有CTA："
            f"{yes_value:,.{decimals}f}"
            f"{unit}"
        ),

        html.P(
            f"沒有CTA："
            f"{no_value:,.{decimals}f}"
            f"{unit}"
        ),

        html.Strong(
            f"差異："
            f"{difference:+,.{decimals}f}"
            f"{unit}"
        )
    ]


def make_bar(
    data,
    value_column,
    title,
    y_title
):
    figure = px.bar(
        data,
        x="CTA",
        y=value_column,
        color="CTA",
        text=value_column,
        title=title,

        category_orders={
            "CTA": [
                "沒有 CTA",
                "有 CTA"
            ]
        },

        color_discrete_map={
            "沒有 CTA": "#9E9E9E",
            "有 CTA": "#4C78A8"
        }
    )

    figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    figure.update_layout(
        xaxis_title="",
        yaxis_title=y_title,
        showlegend=False
    )

    return figure


# ============================================================
# 10. Dashboard互動功能
# ============================================================

@app.callback(
    Output(
        "reach-card",
        "children"
    ),

    Output(
        "engagement-card",
        "children"
    ),

    Output(
        "followers-card",
        "children"
    ),

    Output(
        "reach-chart",
        "figure"
    ),

    Output(
        "engagement-chart",
        "figure"
    ),

    Output(
        "followers-chart",
        "figure"
    ),

    Output(
        "difference-heatmap",
        "figure"
    ),

    Output(
        "consistency-chart",
        "figure"
    ),

    Output(
        "dynamic-insight",
        "children"
    ),

    Input(
        "media-filter",
        "value"
    ),

    Input(
        "category-filter",
        "value"
    ),

    Input(
        "heatmap-metric",
        "value"
    )
)
def update_dashboard(
    selected_media,
    selected_category,
    selected_metric
):
    # 篩選原始資料
    filtered = df.copy()

    if selected_media != "ALL":
        filtered = filtered.loc[
            filtered["media_type"]
            == selected_media
        ]

    if selected_category != "ALL":
        filtered = filtered.loc[
            filtered["content_category"]
            == selected_category
        ]

    # 計算有無CTA的平均值
    summary = (
        filtered.groupby(
            "CTA",
            observed=True
        )
        .agg(
            posts=("post_id", "count"),
            avg_reach=("reach", "mean"),
            avg_engagement_rate=(
                "engagement_rate",
                "mean"
            ),
            avg_followers_gained=(
                "followers_gained",
                "mean"
            )
        )
        .reset_index()
    )

    summary["avg_engagement_rate_pct"] = (
        summary["avg_engagement_rate"]
        * 100
    )

    lookup = summary.set_index("CTA")

    yes_reach = float(
        lookup.loc[
            "有 CTA",
            "avg_reach"
        ]
    )

    no_reach = float(
        lookup.loc[
            "沒有 CTA",
            "avg_reach"
        ]
    )

    yes_engagement = float(
        lookup.loc[
            "有 CTA",
            "avg_engagement_rate_pct"
        ]
    )

    no_engagement = float(
        lookup.loc[
            "沒有 CTA",
            "avg_engagement_rate_pct"
        ]
    )

    yes_followers = float(
        lookup.loc[
            "有 CTA",
            "avg_followers_gained"
        ]
    )

    no_followers = float(
        lookup.loc[
            "沒有 CTA",
            "avg_followers_gained"
        ]
    )

    # KPI卡片
    reach_card = make_card(
        "平均觸及",
        yes_reach,
        no_reach
    )

    engagement_card = make_card(
        "平均互動率",
        yes_engagement,
        no_engagement,
        unit="%",
        decimals=3
    )

    followers_card = make_card(
        "平均新增追蹤者",
        yes_followers,
        no_followers
    )

    # 三張長條圖
    reach_figure = make_bar(
        summary,
        "avg_reach",
        "平均觸及",
        "平均觸及"
    )

    engagement_figure = make_bar(
        summary,
        "avg_engagement_rate_pct",
        "平均互動率",
        "平均互動率（%）"
    )

    followers_figure = make_bar(
        summary,
        "avg_followers_gained",
        "平均新增追蹤者",
        "平均新增追蹤者"
    )

    # 篩選情境差異
    selected_diff = content_diff.copy()

    if selected_media != "ALL":
        selected_diff = selected_diff.loc[
            selected_diff["media_type"]
            == selected_media
        ]

    if selected_category != "ALL":
        selected_diff = selected_diff.loc[
            selected_diff["content_category"]
            == selected_category
        ]

    heatmap_settings = {
        "reach_diff": (
            "CTA平均觸及差異",
            "觸及差異",
            ".1f"
        ),

        "engagement_diff_pp": (
            "CTA平均互動率差異",
            "百分點差異",
            ".3f"
        ),

        "followers_diff": (
            "CTA平均新增追蹤者差異",
            "追蹤者差異",
            ".1f"
        )
    }

    (
        heatmap_title,
        color_title,
        number_format
    ) = heatmap_settings[selected_metric]

    heatmap_data = selected_diff.pivot(
        index="content_category",
        columns="media_type",
        values=selected_metric
    )

    heatmap_figure = px.imshow(
        heatmap_data,
        text_auto=number_format,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=heatmap_title,

        labels={
            "x": "帖子形式",
            "y": "內容主題",
            "color": color_title
        }
    )

    heatmap_figure.update_xaxes(
        side="top"
    )

    # 篩選帳號一致性
    selected_consistency = (
        account_consistency.copy()
    )

    if selected_media != "ALL":
        selected_consistency = (
            selected_consistency.loc[
                selected_consistency["media_type"]
                == selected_media
            ]
        )

    if selected_category != "ALL":
        selected_consistency = (
            selected_consistency.loc[
                selected_consistency[
                    "content_category"
                ]
                == selected_category
            ]
        )

    consistency_long = (
        selected_consistency.melt(
            id_vars=[
                "media_type",
                "content_category",
                "paired_accounts"
            ],

            value_vars=[
                "reach_positive_accounts",
                "engagement_positive_accounts",
                "followers_positive_accounts"
            ],

            var_name="metric",
            value_name="positive_accounts"
        )
    )

    consistency_long[
        "positive_percentage"
    ] = (
        consistency_long[
            "positive_accounts"
        ]
        / consistency_long[
            "paired_accounts"
        ]
        * 100
    )

    consistency_long["metric"] = (
        consistency_long["metric"].map({
            "reach_positive_accounts":
                "觸及",

            "engagement_positive_accounts":
                "互動率",

            "followers_positive_accounts":
                "新增追蹤者"
        })
    )

    consistency_figure = px.bar(
        consistency_long,
        x="content_category",
        y="positive_percentage",
        color="metric",
        barmode="group",
        facet_col="media_type",
        title="有CTA表現較高的帳號比例",

        labels={
            "content_category":
                "內容主題",

            "positive_percentage":
                "帳號比例（%）",

            "metric":
                "指標"
        }
    )

    consistency_figure.add_hline(
        y=50,
        line_dash="dash",
        line_color="black"
    )

    consistency_figure.update_yaxes(
        range=[0, 100]
    )

    consistency_figure.update_xaxes(
        tickangle=45
    )

    consistency_figure.for_each_annotation(
        lambda annotation:
        annotation.update(
            text=annotation.text
            .split("=")[-1]
            .title()
        )
    )

    # 動態文字結論
    media_text = (
        "全部形式"
        if selected_media == "ALL"
        else selected_media.title()
    )

    category_text = (
        "全部主題"
        if selected_category == "ALL"
        else selected_category
    )

    insight = html.P([
        f"目前篩選："
        f"{media_text}、"
        f"{category_text}。",

        html.Br(),

        f"有CTA相較沒有CTA，"
        f"平均觸及差異為"
        f"{yes_reach - no_reach:+,.2f}；",

        f"平均互動率差異為"
        f"{yes_engagement - no_engagement:+.3f}"
        f"個百分點；",

        f"平均新增追蹤者差異為"
        f"{yes_followers - no_followers:+,.2f}。",

        html.Br(),

        "以上為歷史資料中的關聯，"
        "不能直接解釋為CTA造成的效果。"
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
        insight
    )


# ============================================================
# 11. 本機啟動
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)