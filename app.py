from pathlib import Path
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px


# =========================
# 1. 讀取分析結果
# =========================
BASE_DIR = Path(__file__).resolve().parent

controlled = pd.read_csv(
    BASE_DIR / "Q2_controlled_comparison.csv"
)

content_diff = pd.read_csv(
    BASE_DIR / "Q2_content_difference.csv"
)

account_consistency = pd.read_csv(
    BASE_DIR / "Q2_account_consistency.csv"
)


# 將 CTA 欄位統一轉成 True / False
controlled["has_call_to_action"] = (
    controlled["has_call_to_action"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

controlled["CTA"] = np.where(
    controlled["has_call_to_action"],
    "有 CTA",
    "沒有 CTA"
)


# =========================
# 2. 篩選器選項
# =========================

media_options = [
    {"label": "全部形式", "value": "ALL"}
] + [
    {"label": value.title(), "value": value}
    for value in sorted(controlled["media_type"].unique())
]

category_options = [
    {"label": "全部主題", "value": "ALL"}
] + [
    {"label": value, "value": value}
    for value in sorted(controlled["content_category"].unique())
]


# =========================
# 3. 頁面樣式
# =========================

CARD_STYLE = {
    "backgroundColor": "white",
    "padding": "20px",
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
}

SECTION_STYLE = {
    "backgroundColor": "white",
    "padding": "20px",
    "marginTop": "20px",
    "borderRadius": "12px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"
}


# =========================
# 4. 建立 Dash
# =========================

app = Dash(__name__)
server = app.server

app.title = "Q2 CTA Strategy Dashboard"

app.layout = html.Div(
    style={
        "backgroundColor": "#F4F6F8",
        "minHeight": "100vh",
        "padding": "30px",
        "fontFamily": "Arial, sans-serif"
    },
    children=[
        html.H1(
            "Q2 CTA Strategy Dashboard",
            style={"marginBottom": "5px"}
        ),

        html.P(
            "比較不同帖子形式與內容主題中，有無 CTA 的貼文表現。",
            style={
                "color": "#555",
                "marginBottom": "25px"
            }
        ),

        # 篩選器
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "20px"
            },
            children=[
                html.Div([
                    html.Label("帖子形式"),
                    dcc.Dropdown(
                        id="media-filter",
                        options=media_options,
                        value="ALL",
                        clearable=False
                    )
                ]),

                html.Div([
                    html.Label("內容主題"),
                    dcc.Dropdown(
                        id="category-filter",
                        options=category_options,
                        value="ALL",
                        clearable=False
                    )
                ])
            ]
        ),

        # KPI 指標卡
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, 1fr)",
                "gap": "20px",
                "marginTop": "25px"
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

        # 整體比較圖
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("有無 CTA 的平均表現"),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(3, 1fr)",
                        "gap": "10px"
                    },
                    children=[
                        dcc.Graph(id="reach-chart"),
                        dcc.Graph(id="engagement-chart"),
                        dcc.Graph(id="followers-chart")
                    ]
                )
            ]
        ),

        # 差異熱圖
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("不同內容情境的 CTA 差異"),

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
                    inline=True,
                    style={"marginBottom": "15px"}
                ),

                dcc.Graph(id="difference-heatmap")
            ]
        ),

        # 帳號一致性
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("不同帳號中的 CTA 表現一致性"),

                html.P(
                    "黑色虛線代表50%。高於虛線表示超過一半的帳號在使用CTA時表現較高。"
                ),

                dcc.Graph(id="consistency-chart")
            ]
        ),

        # 動態結論
        html.Div(
            style=SECTION_STYLE,
            children=[
                html.H2("目前篩選條件的結果"),
                html.Div(id="dynamic-insight")
            ]
        )
    ]
)


# =========================
# 5. 輔助函數
# =========================

def weighted_summary(data):

    rows = []

    for cta_label, group in data.groupby(
        "CTA",
        observed=True
    ):
        total_posts = group["posts"].sum()

        rows.append({
            "CTA": cta_label,
            "posts": total_posts,

            "avg_reach": np.average(
                group["avg_reach"],
                weights=group["posts"]
            ),

            "avg_engagement_rate": np.average(
                group["avg_engagement_rate"],
                weights=group["posts"]
            ),

            "avg_followers_gained": np.average(
                group["avg_followers_gained"],
                weights=group["posts"]
            )
        })

    result = pd.DataFrame(rows)

    result["avg_engagement_rate_pct"] = (
        result["avg_engagement_rate"] * 100
    )

    return result


def create_card(title, yes_value, no_value, unit="", decimals=2):

    difference = yes_value - no_value
    sign = "+" if difference > 0 else ""

    return [
        html.H3(
            title,
            style={"marginTop": "0"}
        ),

        html.P(
            f"有 CTA：{yes_value:,.{decimals}f}{unit}"
        ),

        html.P(
            f"沒有 CTA：{no_value:,.{decimals}f}{unit}"
        ),

        html.Strong(
            f"差異：{sign}{difference:,.{decimals}f}{unit}"
        )
    ]


def create_bar_chart(data, column, title, y_title):

    fig = px.bar(
        data,
        x="CTA",
        y=column,
        color="CTA",
        text=column,
        title=title,
        category_orders={
            "CTA": ["沒有 CTA", "有 CTA"]
        },
        color_discrete_map={
            "沒有 CTA": "#9E9E9E",
            "有 CTA": "#4C78A8"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title=y_title,
        showlegend=False,
        margin={
            "l": 40,
            "r": 20,
            "t": 60,
            "b": 40
        }
    )

    return fig


# =========================
# 6. Dashboard 互動功能
# =========================

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
    Input("heatmap-metric", "value")
)
def update_dashboard(
    selected_media,
    selected_category,
    selected_metric
):

    # -------------------------
    # 篩選 controlled
    # -------------------------

    filtered = controlled.copy()

    if selected_media != "ALL":
        filtered = filtered[
            filtered["media_type"] == selected_media
        ]

    if selected_category != "ALL":
        filtered = filtered[
            filtered["content_category"]
            == selected_category
        ]

    summary = weighted_summary(filtered)

    lookup = summary.set_index("CTA")

    yes_reach = lookup.loc["有 CTA", "avg_reach"]
    no_reach = lookup.loc["沒有 CTA", "avg_reach"]

    yes_engagement = lookup.loc[
        "有 CTA",
        "avg_engagement_rate_pct"
    ]
    no_engagement = lookup.loc[
        "沒有 CTA",
        "avg_engagement_rate_pct"
    ]

    yes_followers = lookup.loc[
        "有 CTA",
        "avg_followers_gained"
    ]
    no_followers = lookup.loc[
        "沒有 CTA",
        "avg_followers_gained"
    ]

    # -------------------------
    # KPI 卡片
    # -------------------------

    reach_card = create_card(
        "平均觸及",
        yes_reach,
        no_reach
    )

    engagement_card = create_card(
        "平均互動率",
        yes_engagement,
        no_engagement,
        unit="%"
    )

    followers_card = create_card(
        "平均新增追蹤者",
        yes_followers,
        no_followers
    )

    # -------------------------
    # 三張長條圖
    # -------------------------

    reach_fig = create_bar_chart(
        summary,
        "avg_reach",
        "平均觸及",
        "平均觸及"
    )

    engagement_fig = create_bar_chart(
        summary,
        "avg_engagement_rate_pct",
        "平均互動率",
        "平均互動率（%）"
    )

    followers_fig = create_bar_chart(
        summary,
        "avg_followers_gained",
        "平均新增追蹤者",
        "平均新增追蹤者"
    )

    # -------------------------
    # CTA 差異熱圖
    # -------------------------

    diff_filtered = content_diff.copy()

    if selected_media != "ALL":
        diff_filtered = diff_filtered[
            diff_filtered["media_type"]
            == selected_media
        ]

    if selected_category != "ALL":
        diff_filtered = diff_filtered[
            diff_filtered["content_category"]
            == selected_category
        ]

    metric_settings = {
        "reach_diff": {
            "title": "CTA平均觸及差異",
            "color": "觸及差異",
            "format": ".1f"
        },
        "engagement_diff_pp": {
            "title": "CTA平均互動率差異",
            "color": "百分點差異",
            "format": ".3f"
        },
        "followers_diff": {
            "title": "CTA平均新增追蹤者差異",
            "color": "追蹤者差異",
            "format": ".1f"
        }
    }

    setting = metric_settings[selected_metric]

    matrix = diff_filtered.pivot(
        index="content_category",
        columns="media_type",
        values=selected_metric
    )

    heatmap_fig = px.imshow(
        matrix,
        text_auto=setting["format"],
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=setting["title"],
        labels={
            "x": "帖子形式",
            "y": "內容主題",
            "color": setting["color"]
        }
    )

    heatmap_fig.update_xaxes(side="top")

    # -------------------------
    # 帳號一致性圖
    # -------------------------

    consistency_filtered = account_consistency.copy()

    if selected_media != "ALL":
        consistency_filtered = consistency_filtered[
            consistency_filtered["media_type"]
            == selected_media
        ]

    if selected_category != "ALL":
        consistency_filtered = consistency_filtered[
            consistency_filtered["content_category"]
            == selected_category
        ]

    consistency_long = consistency_filtered.melt(
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

    consistency_long["positive_percentage"] = (
        consistency_long["positive_accounts"]
        / consistency_long["paired_accounts"]
        * 100
    )

    consistency_long["metric"] = (
        consistency_long["metric"].map({
            "reach_positive_accounts": "觸及",
            "engagement_positive_accounts": "互動率",
            "followers_positive_accounts": "新增追蹤者"
        })
    )

    consistency_fig = px.bar(
        consistency_long,
        x="content_category",
        y="positive_percentage",
        color="metric",
        barmode="group",
        facet_col="media_type",
        title="有 CTA 表現較高的帳號比例",
        labels={
            "content_category": "內容主題",
            "positive_percentage": "帳號比例（%）",
            "metric": "指標"
        }
    )

    consistency_fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="black"
    )

    consistency_fig.update_yaxes(
        range=[0, 100]
    )

    consistency_fig.update_xaxes(
        tickangle=45
    )

    # -------------------------
    # 動態文字結論
    # -------------------------

    reach_difference = yes_reach - no_reach
    engagement_difference = (
        yes_engagement - no_engagement
    )
    followers_difference = (
        yes_followers - no_followers
    )

    media_text = (
        "全部帖子形式"
        if selected_media == "ALL"
        else selected_media.title()
    )

    category_text = (
        "全部內容主題"
        if selected_category == "ALL"
        else selected_category
    )

    insight = html.P(
        [
            f"目前篩選：{media_text}、{category_text}。",

            html.Br(),

            f"有 CTA 相較沒有 CTA，平均觸及差異為 "
            f"{reach_difference:+,.2f}；",

            f"平均互動率差異為 "
            f"{engagement_difference:+.3f} 個百分點；",

            f"平均新增追蹤者差異為 "
            f"{followers_difference:+,.2f}。",

            html.Br(),

            "以上結果代表歷史資料中的關聯，不能直接解釋為 CTA 造成的效果。"
        ]
    )

    return (
        reach_card,
        engagement_card,
        followers_card,
        reach_fig,
        engagement_fig,
        followers_fig,
        heatmap_fig,
        consistency_fig,
        insight
    )


# =========================
# 7. 啟動 Dashboard
# =========================

if __name__ == "__main__":
    app.run(debug=True)
