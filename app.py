import pandas as pd
import pydeck as pdk
import streamlit as st


st.set_page_config(page_title="Swiggy Sales Analytics", layout="wide")


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows (same order line: date, restaurant, dish, price)."""
    key = ["Order Date", "Restaurant Name", "Dish Name", "Price (INR)"]
    return df.drop_duplicates(subset=[c for c in key if c in df.columns])


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with out-of-range or IQR-based outliers."""
    out = df.copy()

    # Ensure numeric
    for col in ["Price (INR)", "Rating", "Rating Count"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Price: keep only positive; then remove IQR outliers
    if "Price (INR)" in out.columns:
        out = out[out["Price (INR)"].notna() & (out["Price (INR)"] > 0)]
        q1, q3 = out["Price (INR)"].quantile(0.25), out["Price (INR)"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = out[(out["Price (INR)"] >= lo) & (out["Price (INR)"] <= hi)]

    # Rating: valid range 0–5 (drop invalid; keep NaN if present)
    if "Rating" in out.columns:
        out = out[out["Rating"].isna() | ((out["Rating"] >= 0) & (out["Rating"] <= 5))]

    # Rating Count: remove IQR outliers (non-negative)
    if "Rating Count" in out.columns:
        out = out[out["Rating Count"].notna() & (out["Rating Count"] >= 0)]
        q1, q3 = out["Rating Count"].quantile(0.25), out["Rating Count"].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            hi = q3 + 1.5 * iqr
            out = out[out["Rating Count"] <= hi]

    return out


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and prepare the Swiggy sales data (duplicates and outliers removed)."""
    df = pd.read_csv("sweggy.csv", encoding="latin1")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"], format="%d-%m-%y", errors="coerce"
    )
    df = df.dropna(subset=["Order Date"])

    # Exclude 22 Feb 2025 (known outlier date)
    exclude_date = pd.Timestamp("2025-02-22")
    df = df[df["Order Date"].dt.normalize() != exclude_date]

    df = _remove_duplicates(df)
    df = _remove_outliers(df)

    df["Sales"] = df["Price (INR)"]

    # Derive food type (Veg / Non Veg) from category and dish name heuristically
    def classify_food_type(row: pd.Series) -> str:
        text = f"{row.get('Category', '')} {row.get('Dish Name', '')}".lower()
        nonveg_keywords = [
            "chicken",
            "mutton",
            "egg",
            "fish",
            "prawn",
            "meat",
            "non veg",
            "non-veg",
            "bacon",
        ]
        if any(kw in text for kw in nonveg_keywords):
            return "Non Veg"
        return "Veg"

    df["Food Type"] = df.apply(classify_food_type, axis=1)

    # Time-related helpers
    df["Date"] = df["Order Date"].dt.date
    df["Week"] = df["Order Date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)

    return df


df = load_data()

st.title("Swiggy Sales Analytics Dashboard")
st.write(
    "Comprehensive view of Swiggy sales performance with key KPIs, trends, "
    "food type split, geography, and time-based summaries. "
    "**Data cleaned:** duplicate rows and outliers removed; 22 Feb 2025 excluded."
)


# =========================
# Top-level KPI cards
# =========================
total_sales = float(df["Sales"].sum())
avg_rating = float(df["Rating"].mean()) if not df["Rating"].isna().all() else 0.0
total_orders = int(len(df))
rating_count = int(df["Rating Count"].sum())
avg_order_value = total_sales / total_orders if total_orders > 0 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Sales", f"₹{total_sales:,.0f}")

with col2:
    st.metric("Average Rating", f"{avg_rating:.2f}")

with col3:
    st.metric("Average Order Value", f"₹{avg_order_value:,.0f}")

with col4:
    st.metric("Rating Count", f"{rating_count:,}")

with col5:
    st.metric("Total Orders", f"{total_orders:,}")


# =========================
# Tabs for detailed views
# =========================
tab_trends, tab_food_geo, tab_summary = st.tabs(
    [
        "Trends (Monthly / Daily / Weekly)",
        "Food Type & State Map",
        "Quarterly & Top Cities",
    ]
)


# -------------------------
# Trends tab
# -------------------------
with tab_trends:
    st.subheader("Monthly, Daily and Weekly Sales Trends")

    # Monthly Sales Trend
    monthly = (
        df.groupby("Month")["Sales"].sum().reset_index().sort_values("Month")
    )

    # Daily Sales Trend
    daily = (
        df.groupby("Date")["Sales"].sum().reset_index().sort_values("Date")
    )

    # Weekly Sales Trend
    weekly = (
        df.groupby("Week")["Sales"].sum().reset_index().sort_values("Week")
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Monthly Sales Trend**")
        if not monthly.empty:
            st.line_chart(monthly.set_index("Month")["Sales"])
        else:
            st.info("No monthly data available.")

    with col_m2:
        st.markdown("**Daily Sales Trend**")
        if not daily.empty:
            st.line_chart(daily.set_index("Date")["Sales"])
        else:
            st.info("No daily data available.")

    st.markdown("**Weekly Trend Analysis**")
    if not weekly.empty:
        st.line_chart(weekly.set_index("Week")["Sales"])
    else:
        st.info("No weekly data available.")


# -------------------------
# Food type & geography tab
# -------------------------
with tab_food_geo:
    col_f1, col_f2 = st.columns(2)

    # Total Sales Trend by Food Type (Veg / Non Veg) over time
    with col_f1:
        st.subheader("Total Sales Trend by Food Type (Veg vs Non Veg)")
        monthly_food = (
            df.groupby(["Month", "Food Type"])["Sales"]
            .sum()
            .reset_index()
            .sort_values("Month")
        )

        if not monthly_food.empty:
            pivot_food = monthly_food.pivot(
                index="Month", columns="Food Type", values="Sales"
            ).fillna(0)
            st.line_chart(pivot_food)
        else:
            st.info("No data available for food-type trend.")

    # Total Sales by State (Map Visualization)
    with col_f2:
        st.subheader("Total Sales by State (Map)")

        state_sales = (
            df.groupby("State")["Sales"].sum().reset_index()
        )

        # Approximate coordinates for Indian states (extend as needed)
        state_coords = {
            "Karnataka": {"lat": 15.3173, "lon": 75.7139},
            "Maharashtra": {"lat": 19.7515, "lon": 75.7139},
            "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569},
            "Telangana": {"lat": 17.1232, "lon": 79.2088},
            "Delhi": {"lat": 28.7041, "lon": 77.1025},
            "West Bengal": {"lat": 22.9868, "lon": 87.8550},
            "Gujarat": {"lat": 22.2587, "lon": 71.1924},
            "Rajasthan": {"lat": 27.0238, "lon": 74.2179},
        }

        state_map_df = state_sales[
            state_sales["State"].isin(state_coords.keys())
        ].copy()

        if not state_map_df.empty:
            state_map_df["lat"] = state_map_df["State"].map(
                lambda s: state_coords[s]["lat"]
            )
            state_map_df["lon"] = state_map_df["State"].map(
                lambda s: state_coords[s]["lon"]
            )

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=state_map_df,
                get_position="[lon, lat]",
                get_radius="Sales / 10",
                get_fill_color=[255, 99, 71, 180],
                pickable=True,
            )

            view_state = pdk.ViewState(
                latitude=20.5937,
                longitude=78.9629,
                zoom=4,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{State}\nSales: ₹{Sales}"},
                )
            )
        else:
            st.info(
                "No recognizable state names found for map visualization. "
                "Ensure the `State` column contains valid Indian state names."
            )


# -------------------------
# Quarterly performance & top cities tab
# -------------------------
with tab_summary:
    st.subheader("Quarterly Performance Summary")

    quarterly = (
        df.groupby("Quarter")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Orders=("Sales", "count"),
            Avg_Rating=("Rating", "mean"),
        )
        .reset_index()
        .sort_values("Quarter")
    )

    if not quarterly.empty:
        st.dataframe(
            quarterly.style.format(
                {
                    "Total_Sales": "₹{:.0f}",
                    "Total_Orders": "{:,.0f}",
                    "Avg_Rating": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("**Quarterly Sales Trend**")
        st.bar_chart(
            quarterly.set_index("Quarter")["Total_Sales"]
        )
    else:
        st.info("No quarterly data available.")

    st.subheader("Top 5 Cities by Sales")

    city_sales = (
        df.groupby("City")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Sales", ascending=False)
        .head(5)
    )

    if not city_sales.empty:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.bar_chart(
                city_sales.set_index("City")["Sales"]
            )

        with col_c2:
            st.dataframe(
                city_sales.style.format({"Sales": "₹{:.0f}"}),
                use_container_width=True,
            )
    else:
        st.info("No city-level data available.")

