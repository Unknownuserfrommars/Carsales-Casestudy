import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu as om
import plotly.io as pio
import math
from scipy import stats
# from pandas.api.types import is_categorical_dtype, is_bool_dtype, is_numeric_dtype, is_object_dtype

pio.renderers.default = 'browser'
st.set_page_config(layout="wide")
pcq = px.colors.qualitative

my_path = "C:\\Users\\Kevin\\OneDrive\\Desktop\\carsalesreport\\archive\\"
bmw = pd.read_csv(my_path + 'BMW_Car_Sales_Classification.csv')

bmw.columns = bmw.columns.str.lower().str.strip()

# TODO: Make a list of "Kevin's Colours" x 30, pass list in to every plot as in the colour discrete sequence arg.

# bmw["model_norm"] = bmw["model"].str.replace(r"(\d+)\s+Series", r"S\1", regex=True)
# Age = 2025 - Year
bmw['car_age'] = 2025 - bmw['year']
# Mileage_per_year = Mileage_KM / max(Age,1)
bmw['mileage_per_year'] = bmw['mileage_km'] / np.maximum(2025 - bmw['year'], 1)
# Price_per_Litre = Price_USD / Engine_Size_L
bmw['price_per_litre'] = bmw['price_usd'] / bmw['engine_size_l']
# Price_per_100KM = Price_USD / (Mileage_KM/100)
bmw['price_per_100km'] = bmw['price_usd'] / (bmw['mileage_km']/100)
# Price_per_1kKM = Price_USD / (Mileage_KM/1000)
# bmw['price_per_1kkm'] = bmw['price_usd'] / (bmw['mileage_km']/1000)

# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# X = bmw[['engine_size_l', 'mileage_km', 'price_usd', 'car_age', 'mileage_per_year', 'price_per_litre', 'price_per_100km', 'price_per_1kkm']]
# y = bmw['sales_classification']
# model.fit(X, y)

cat_dict = {
    "model":"Car Model",
    "region":"Global Region",
    "color":"Color",
    "fuel_type":"Fuel Type",
    "transmission":"Transmission",
    "sales_classification":"Sales Volume Level",
    }

num_dict = {
    "year":"Year",
    "engine_size_l":"Engine Size (Liters)",
    "mileage_km":"Milage (Km)",
    "price_usd":"Price ($)",
    "sales_volume":"Sales Count",
    "car_age":"Car Age (Years)",
    "mileage_per_year":"Mileage per Year (Km)",
    "price_per_litre":"Price per Litre ($/L)",
    "price_per_100km":"Price per 100 Km ($/100Km)",
    #"price_per_1kkm":"Price per 1,000 Km ($/1,000Km)",
    }

all_dict = num_dict.copy()
all_dict.update(cat_dict)


def predictors_to_probabilities(
    df: pd.DataFrame,
    target_col: str = "sales_classification",
    epsilon: float = 1e-8,
    cat_suffix: str = "_prob",
    num_suffix: str = "_cdf",
) -> tuple[pd.DataFrame, dict]:
    """
    Convert all predictor columns (exclude `target_col`) to probability-like values:
      - Categorical -> frequency (float in [0,1])
      - Numeric     -> Normal CDF fitted per column

    Returns (transformed_df, fitted_stats)
      fitted_stats = {
        "categorical": {col: {category: freq, ...}, ...},
        "numeric":     {col: {"mean": m, "std": s}, ...}
      }
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col]).copy()

    # cat_cols, num_cols = [], []
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=np.number).columns

    fitted = {"categorical": {}, "numeric": {}}
    out = pd.DataFrame(index=X.index)

    # Categoricals → frequency probabilities
    for col in cat_cols:
        freqs = X[col].value_counts(normalize=True, dropna=False).to_dict()
        fitted["categorical"][col] = freqs
        out[col + cat_suffix] = X[col].map(freqs).astype(float)

    # Numerics → Normal CDF
    for col in num_cols:
        m = float(np.mean(X[col].values))
        s = float(np.std(X[col].values, ddof=0))
        if s < epsilon:
            s = epsilon
        fitted["numeric"][col] = {"mean": m, "std": s}
        out[col + num_suffix] = stats.norm.cdf((X[col].values - m) / s)

    return out, fitted

# x, _ = predictors_to_probabilities(bmw, target_col='sales_classification')
# print(x.columns.str.endswith('_cdf'))

# Do this code with the final df:
# x[x.columns[x.columns.str.endswith('_cdf')]].prod(axis=1)

class NullPointerException(ValueError):
    pass

# DONE: Overall func takes in 2 args: a df (inc. only the predictors u want, use multiselect) + a target variable and the string of the target variable. 1. Seperate out the target variable, determine which ones are numeric and category -> cat_vars and num_vars; 2. Create 2 seperate dfs (eg df_high & df_low) based on target variable for diff. scenarios; 3. Calculate the priors for the high and the low; 4. Create 2 empty dicts for the df_high and df_low probs; 5. Loop thru all columns in the df - the target variable (obviously): check to see if col is in numeric/category -> if category do .value_counts() (With the high create the dict based on df_high; with the low create the dict based on df_low) and do the mean and std for both and store the results into the corresponding dict with key of dict = column name; 6. Use pd.DataFrame() to convert to a dataframe and for each row have 2 columns high_probs and low_probs...

def custom_naive_bayes(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    df: DataFrame that already contains the target column
    target: string name of the target column
    Returns: DataFrame with columns [prob_high, prob_low, actual, predicted]
    """

    # Assert if DataFrame has NaN values
    assert df.isna().sum(axis=0).sum() == 0, "DataFrame contains NaN values. Please handle them before passing to this function."

    # Step 1
    y = df[target]
    X = df.drop(columns=[target])
    cat_vars = []
    num_vars = []

    # for col, dtype in X.dtypes.items():
    #     if pd.api.types.is_numeric_dtype(dtype):
    #         num_vars.append(col)
    #     else:
    #         cat_vars.append(col)

    cat_vars = X.select_dtypes(include=['object']).columns
    num_vars = X.select_dtypes(include=np.number).columns

    # Step 2 - sorting the high and low dataframes
    df_high = df[df[target] == "High"]
    df_low  = df[df[target] == "Low"]

    # Step 3 - calculating the priors
    prior_high = len(df_high) / len(df)
    prior_low  = len(df_low)  / len(df)
    
    # Step 4
    prob_high, prob_low = {}, {}

    # Step 5
    for col in X.columns:
        if col in cat_vars:
            vals_high = df_high[col].value_counts(normalize=True)
            vals_low  = df_low[col].value_counts(normalize=True)
            V = len(df[col].unique())
            # high_dict[col] = (vals_high + 1) / (len(df_high) + V)
            # low_dict[col]  = (vals_low  + 1) / (len(df_low)  + V)
            prob_high[col] = df[col].map(vals_high).fillna(1e-8).astype(float)
            prob_low[col]  = df[col].map(vals_low).fillna(1e-8).astype(float)
        elif col in num_vars:
            # vals_high[col] = (df_high[col].mean(), df_high[col].std(ddof=0))
            # vals_low[col] = (df_low[col].mean(),  df_low[col].std(ddof=0))
            hi_mean = df_high[col].mean()
            hi_std  = df_high[col].std(ddof=0)
            lo_mean = df_low[col].mean()
            lo_std  = df_low[col].std(ddof=0)
            vals_high = stats.norm(hi_mean, hi_std)
            vals_low  = stats.norm(lo_mean, lo_std)
            prob_high[col] = vals_high.pdf(df[col])
            prob_low[col]  = vals_low.pdf(df[col])
        else:
            raise NullPointerException(f"Column '{col}' is neither categorical nor numeric. An error occured.")

    df_high = pd.DataFrame(prob_high).prod(axis=1) * prior_high
    df_low  = pd.DataFrame(prob_low).prod(axis=1) * prior_low

    preds = np.where(df_high > df_low, "High", "Low")

    # def gauss(x, mu, sigma):
    #     sigma = max(sigma, 1e-6)
    #     return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # for _, row in X.iterrows():
    #     p_high = prior_high
    #     p_low  = prior_low
    #     for col in X.columns:
    #         if col in cat_vars:
    #             v = row[col]
    #             p_high *= high_dict[col].get(v, 1 / (len(df_high) + len(high_dict[col])))
    #             p_low  *= low_dict[col].get(v,  1 / (len(df_low)  + len(low_dict[col])))
    #         else:
    #             mu_h, sd_h = high_dict[col]
    #             mu_l, sd_l = low_dict[col]
    #             p_high *= gauss(row[col], mu_h, sd_h)
    #             p_low  *= gauss(row[col], mu_l, sd_l)
    #     norm = p_high + p_low
    #     prob_high.append(p_high / norm if norm else 0)
    #     prob_low.append(p_low / norm if norm else 0)

    # Step 6

    # out = pd.DataFrame({
    #     "prob_high": prob_high,
    #     "prob_low": prob_low,
    #     "actual": y.values,
    #     "predicted": ["high" if h >= l else "low" for h, l in zip(prob_high, prob_low)]
    # })

    out = pd.DataFrame({
        "prob_high": df_high.map(lambda v: f'{v:2e}'),
        "prob_low": df_low.map(lambda v: f'{v:2e}'),
        "actual": y,
        "predicted": preds,
    })

    accuracy = np.mean(out["actual"] == out["predicted"])

    return out, accuracy

all_cols = list(num_dict.keys()) + list(cat_dict.keys())
# res, acc = custom_naive_bayes(bmw[all_cols], 'sales_classification')
# st.write(f"Naive Bayes accuracy: {acc:.2f}")
# st.write(res)

# ---- 
with st.sidebar:
    s = om(
        menu_title = 'The Great Navigation Pane of All Time',
        options = ['Abstract', 'Background Information', 'Data Cleaning','Exploratory', 'Naive Bayes Prediction', 'Analysis', 'ph4', 'ph5', 'Conclusion', 'Bibliography'],
        menu_icon = 'house-door-fill',
        icons = ['person-arms-up', 'basket', 'filetype-csv', 'search', 'microsoft-teams', 'key-fill', 'trophy-fill', 'flag', 'star-fill', 'list-ol'],
        default_index = 0,
        )
# ----


# f = px.histogram()
# f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=2)))

if s == 'Abstract':
    st.title('The Great Navigation Pane of All Time')
    st.text("This is a placeholder. Come back later for the real content.")




if s == 'Background Information':
    st.title('Background Information')
    st.text("This is a placeholder. Come back later for the real content.")






if s == 'Data Cleaning':
    st.title('Data Cleaning')
    st.text("This is a placeholder. Come back later for the real content.")







if s == 'Exploratory':
    st.title('Exploratory Data Analysis')
    st.text("This is a placeholder. Come back later for the real content.")
    
    st.header("Exploring using Sales Volume Level")
    st.subheader("Placeholder - Histogram 2 cat 1 num") # TODO: Placeholder, pls change
    col1, col2 = st.columns([2,3])
    with st.form("Histogram 2 cat 1 num A"):
        fig1_cat = col1.selectbox("Select a category feature to display on the graph", np.setdiff1d(list(cat_dict.values()), "Sales Volume Level"), key=1)
        fig1_y = col1.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=2)
        
        fig1_cat_d = [key for key, value in cat_dict.items() if value == fig1_cat][0]
        fig1_y_d = [key for key, value in num_dict.items() if value == fig1_y][0]
        
        fig1_logy = col1.checkbox("Click if you want log y scale", key=3)
        # fig1_kde = col1.checkbox("Click if you want to overlay a KDE", key="fig1_kde")
        # fig1_norm = col1.checkbox("Click if you want to overlay a normal curve", key="fig1_norm")
        # x_range = np.linspace(bmw[fig1_y_d].min(), bmw[fig1_y_d].max(), 100)

        fig1_color_axis = col1.checkbox("Click if you want to make the category variable the x axis", key="fig1_color_axis")

        submitted = st.form_submit_button("Click to produce the histogram")

        fig1_x_d = "sales_classification"
        fig1_color_d = fig1_cat_d
        if fig1_color_axis:
        #     f1x = "sales_classification"
            fig1_x_d = fig1_cat_d
            fig1_color_d = "sales_classification"
        fig1_x = all_dict[fig1_x_d]
        fig1_color = all_dict[fig1_color_d]

        if submitted:
            fig1 = px.histogram(bmw, x=fig1_x_d, y=fig1_y_d, color=fig1_color_d, labels=all_dict, barmode='group', log_y=fig1_logy, histfunc='avg', title=f'Histogram of comparing {fig1_x} and {fig1_y} by {fig1_color}')
            fig1.update_traces(marker_line_width=1)
            # if fig1_kde:
            #     kde = stats.gaussian_kde(bmw[fig1_y_d])
            #     fig1.add_trace(go.Scatter(x=x_range, y=kde.pdf(x_range), mode='lines', name='KDE', line=dict(color='#66ff66', width=2)))
            # if fig1_norm:
            #     mu, std = bmw[fig1_y_d].mean(), bmw[fig1_y_d].std()
            #     norm_dist = stats.norm.pdf(x_range, mu, std)
            #     fig1.add_trace(go.Scatter(x=x_range, y=norm_dist, mode='lines', name='Normal Curve', line=dict(color='#ff66ff', width=2)))
            col2.plotly_chart(fig1)

    st.markdown("<hr style=\"height: 5px; background-color: red;\" />", unsafe_allow_html=True)

    # st.header("Exploring using Car Model")
    # st.subheader("Placeholder - Histogram 2 cat 1 num") # TODO: Placeholder, pls change
    # col3, col4 = st.columns([2,3])
    # with st.form("Histogram 2 cat 1 num B"):
    #     fig2_x = col3.selectbox("Select a category feature for the x-axis", np.setdiff1d(list(cat_dict.values()), "Car Model"), key=4)
    #     fig2_y = col3.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=5)
    #     fig2_x_d = [key for key, value in cat_dict.items() if value == fig2_x][0]
    #     fig2_y_d = [key for key, value in num_dict.items() if value == fig2_y][0]
    #     fig2_logy = col3.checkbox("Click if you want log y scale", key=6)

    #     submitted = st.form_submit_button("Click to produce the histogram")
    #     if submitted:
    #         fig2 = px.histogram(bmw, x=fig2_x_d, y=fig2_y_d, color="model", labels=all_dict, barmode='group', log_y=fig2_logy, histfunc='avg', title=f'Histogram of comparing {fig2_x} and {fig2_y} by Car Model')
    #         fig2.update_traces(marker_line_width=1)
    #         col4.plotly_chart(fig2)

    # st.header("Exploring using Global Region")
    # st.subheader("Placeholder - Histogram 2 cat 1 num")
    # col5, col6 = st.columns([2,3])
    # with st.form("Histogram 2 cat 1 num C"):
    #     fig3_x = col5.selectbox("Select a category feature for the x-axis", np.setdiff1d(list(cat_dict.values()), "Global Region"), key=7)
    #     fig3_y = col5.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=8)
    #     fig3_x_d = [key for key, value in cat_dict.items() if value == fig3_x][0]
    #     fig3_y_d = [key for key, value in num_dict.items() if value == fig3_y][0]
    #     fig3_logy = col5.checkbox("Click if you want log y scale", key=9)
    
    st.header("Exploring of 1 cat 1 num + KDE/Norm lines") # Placeholder - TODO: Change
    col7, col8 = st.columns([2,3])
    with st.form("Histogram 1 cat 1 num + KDE/Norm lines"):
        fig4_x = col7.selectbox("Select a numeric feature for the x-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=10)

        fig4_x_d = [key for key, value in num_dict.items() if value == fig4_x][0]

        usr_sel_bin_num = col7.checkbox("Do you want to select the number of bins? (Default is 10)", key="usbl4")
        fig4_bins = col7.slider("The amount of bins you want", 5, 35, step=1, key="fig4_bins") if usr_sel_bin_num else 10 

        # fig4_logy = col7.checkbox("Click if you want log y scale", key=11)
        fig4_kde = col7.checkbox("Click if you want to overlay a KDE", key="fig4_kde")
        fig4_norm = col7.checkbox("Click if you want to overlay a normal curve", key="fig4_norm")
        fig4_mean_med = col7.checkbox("Click if you want to overlay the mean and median lines", key="fig4_mean_med")
        fig4_x_range = np.linspace(bmw[fig4_x_d].min(), bmw[fig4_x_d].max(), 100)

        submitted = st.form_submit_button("Click to produce the histogram")
        if submitted:
            fig4 = px.histogram(bmw, x=fig4_x_d, color='sales_classification', labels=all_dict, nbins=fig4_bins, histnorm="probability density", barmode='group', title=f'Histogram of {fig4_x}')  # TODO: Possible check for Color discrete map conflict w/ default. Can: use map assign one color to high; another color to low.
            fig4.update_traces(marker_line_width=1)
            if fig4_kde:
                kde = stats.gaussian_kde(bmw[fig4_x_d])
                fig4.add_trace(go.Scatter(x=fig4_x_range, y=kde.pdf(fig4_x_range), mode='lines', name='KDE', line=dict(color='#66ff66', width=2)))
            if fig4_norm:
                mu, std = bmw[fig4_x_d].mean(), bmw[fig4_x_d].std()
                norm_dist = stats.norm.pdf(fig4_x_range, mu, std)
                fig4.add_trace(go.Scatter(x=fig4_x_range, y=norm_dist, mode='lines', name='Normal Curve', line=dict(color='#ff66ff', width=2)))
            if fig4_mean_med:
                mean = bmw[fig4_x_d].mean()
                median = bmw[fig4_x_d].median()
                annot_pos_mean = 'top left' if mean < median else 'bottom right'
                annot_pos_median = 'top left' if not mean < median else 'bottom right'
                fig4.add_vline(x=mean, line=dict(color='blue', width=2, dash='dash'), name='Mean', annotation_text=f'<b>Mean:<br>{mean:.2f}</b>', annotation_position=annot_pos_mean, annotation_font_color='blue', annotation_font_size=8) # TODO: Chooseable later, use hex color, always make sure the font color = line color TODO: Change color avoid conflict with default color map
                fig4.add_vline(x=median, line=dict(color='orange', width=2, dash='dash'), name='Median', annotation_text=f'<b>Median:<br>{median:.2f}</b>', annotation_position=annot_pos_median, annotation_font_color='orange', annotation_font_size=8) # TODO: Chooseable later, use hex color, always make sure the font color = line color
            
            col8.plotly_chart(fig4)
    
    st.markdown("<hr style=\"height: 5px; background-color: red;\" />", unsafe_allow_html=True)


    st.header("Exploring sunburst - 1 cat 1 num") # Placeholder - TODO: Change
    col9, col10 = st.columns([2,3])
    with st.form("Sunburst 1 cat 1 num"):
        fig5_cat = col9.selectbox("Select a category feature for the sunburst", np.setdiff1d(list(cat_dict.values()), "Sales Volume Level"), key=12)
        fig5_y = col9.selectbox("Select a numeric feature for the values", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=13)
        
        fig5_cat_d = [key for key, value in cat_dict.items() if value == fig5_cat][0]
        fig5_y_d = [key for key, value in num_dict.items() if value == fig5_y][0]

        fig5_color_axis = col9.checkbox("Click if you want the category variable in the center", key="fig5_color_axis")
        fig5_percent_method = col9.selectbox("Select the percentage display method", ["percent entry", "percent parent"], index=0, key="fig5_percent_method")

        # if fig5_color_axis:
        #     fig5_cat_d, fig5_y_d = "sales_classification", fig5_cat_d
        #     fig5_cat, fig5_y = all_dict[fig5_cat_d], all_dict[fig5_y_d]
        
        sb_path = [fig5_cat_d, 'sales_classification'] if fig5_color_axis else ['sales_classification', fig5_cat_d]

        submitted = st.form_submit_button("Click to produce the sunburst")
        # if submitted:
        #     sbdf = bmw.copy()
        #     sbdf[fig5_y_d] = sbdf.groupby(sb_path)[fig5_y_d].transform('mean') # TODO: Challenge: Change every for x: divide the mean on the number of items in the same group.
        #     fig5 = px.sunburst(bmw, path=sb_path, values=fig5_y_d, color='sales_classification', color_discrete_map={'High':'#ff6666', 'Medium':'#66ff66', 'Low':'#6666ff'}, labels=all_dict, title=f'Sunburst of {fig5_cat} and {fig5_y} by Sales Volume Level', height=800, width=800)
        #     col10.plotly_chart(fig5)
        if submitted:
            sbdf = bmw.copy()
            grp = sbdf.groupby(sb_path)[fig5_y_d]
            counts = grp.transform('size')
            means  = grp.transform('mean')
            
            sbdf['_sb_value'] = means / counts
            fig5 = px.sunburst(
                sbdf, path=sb_path, values='_sb_value', color='sales_classification', color_discrete_map={'High':'#ff6666', 'Medium':'#66ff66', 'Low':'#6666ff'}, labels=all_dict, title=f'Sunburst of {fig5_cat} and {fig5_y} by Sales Volume Level', height=800, width=800
            )
            fig5.update_traces(textinfo=f'label+{fig5_percent_method}')
            col10.plotly_chart(fig5)

    
    # st.header("Exploring scatter plot - 2 num + Hardcoded color") # Placeholder - TODO: Change
    # col11, col12 = st.columns([2,3])
    # with st.form("Scatter plot 2 num + Hardcoded color"):
    #     fig6_x = col11.selectbox("Select a numeric feature for the x-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count",]), key=14)
    #     fig6_y = col11.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", fig6_x]), key=15)
        
    #     fig6_x_d = [key for key, value in num_dict.items() if value == fig6_x][0]
    #     fig6_y_d = [key for key, value in num_dict.items() if value == fig6_y][0]


    #     usr_sel_trendline = col11.checkbox("Do you want to add a trendline? (Default is no trendline)", key="usblt6")
    #     fig6_trendline = 'ols' if usr_sel_trendline else None

    #     submitted = st.form_submit_button("Click to produce the scatter plot")
    #     if submitted:
    #         fig6 = px.scatter(bmw, x=fig6_x_d, y=fig6_y_d, color='sales_classification', trendline=fig6_trendline, color_discrete_map={'High':'#ff6666', 'Medium':'#66ff66', 'Low':'#6666ff'}, labels=all_dict, title=f'Scatter plot of {fig6_x} and {fig6_y} by Sales Volume Level', hover_data=['model', 'region', 'year', 'engine_size_l', 'mileage_km', 'price_usd'])
    #         col12.plotly_chart(fig6)

    st.header("Exploring line plot - hardcode x = year, 1 num as y, 2 traces of sales classification as color")
    col11, col12 = st.columns([2,3])
    with st.form("Line plot year 1 num 2 traces"):
        fig6_y = col11.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=14)
        
        fig6_y_d = [key for key, value in num_dict.items() if value == fig6_y][0]

        fig6_logy = col11.checkbox("Click if you want log y scale", key=15)

        lpdf = bmw.groupby(['year', 'sales_classification'])[fig6_y_d].mean().reset_index()

        # st.write(bmw['price_usd'].describe())

        submitted = st.form_submit_button("Click to produce the line plot")
        if submitted:
            fig6 = px.line(lpdf, x='year', y=fig6_y_d, color='sales_classification', color_discrete_map={'High':'#ff6666', 'Medium':'#66ff66', 'Low':'#6666ff'}, labels=all_dict, title=f'Line plot of Year and {fig6_y} by Sales Volume Level', markers=True, log_y=fig6_logy)
            fig6.update_traces(marker_line_width=1)
            col12.plotly_chart(fig6)


    st.header("Histogram 2 num 1 cat") # Placeholder - TODO: Change
    col13, col14 = st.columns([2,3])
    with st.form("Histogram 2 num 1 cat"):
        fig7_x = col13.selectbox("Select a numeric feature for the x-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=16)
        fig7_y = col13.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year", fig7_x]), key=17)
        fig7_cat = col13.selectbox("Select a category feature to display on the graph", np.setdiff1d(list(cat_dict.values()), "Sales Volume Level"), key=18)
        
        fig7_x_d = [key for key, value in num_dict.items() if value == fig7_x][0]
        fig7_y_d = [key for key, value in num_dict.items() if value == fig7_y][0]
        fig7_cat_d = [key for key, value in cat_dict.items() if value == fig7_cat][0]

        usr_sel_bin_num = col13.checkbox("Do you want to select the number of bins? (Default is 10)", key="usbl7")
        fig7_bins = col13.slider("The amount of bins you want", 5, 35, step=1, key="fig7_bins") if usr_sel_bin_num else 10 

        fig7_logy = col13.checkbox("Click if you want log y scale", key=19)

        submitted = st.form_submit_button("Click to produce the histogram")
        if submitted:
            fig7 = px.histogram(bmw, x=fig7_x_d, y=fig7_y_d, color=fig7_cat_d, labels=all_dict, nbins=fig7_bins, barmode='group', log_y=fig7_logy, histfunc='avg', title=f'Histogram of comparing {fig7_x} and {fig7_y} by {fig7_cat}')
            fig7.update_traces(marker_line_width=1)
            col14.plotly_chart(fig7)


    st.header("Exploring box plot - 2 cat 1 num") # Placeholder - TODO: Change
    # User can select whether target var as x or color
    col15, col16 = st.columns([2,3])
    with st.form("Box plot 2 cat 1 num"):
        fig8_cat = col15.selectbox("Select a category feature for the x-axis", np.setdiff1d(list(cat_dict.values()), "Sales Volume Level"), key=20)
        fig8_y = col15.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=21)
        
        fig8_cat_d = [key for key, value in cat_dict.items() if value == fig8_cat][0]
        fig8_y_d = [key for key, value in num_dict.items() if value == fig8_y][0]

        fig8_color_axis = col15.checkbox("Click if you want to make the category variable the color", key="fig8_color_axis")
        fig8_logy = col15.checkbox("Click if you want log y scale", key=22)
        fig8_points = col15.checkbox("Click if you want to show all points", key="fig8_points")

        submitted = st.form_submit_button("Click to produce the box plot")

        fig8_x_d = fig8_cat_d
        fig8_color_d = "sales_classification"
        
        if fig8_color_axis:
            fig8_x_d = "sales_classification"
            fig8_color_d = fig8_cat_d
        fig8_x = all_dict[fig8_x_d]
        fig8_color = all_dict[fig8_color_d]

        if submitted:
            fig8 = px.box(bmw, x=fig8_x_d, y=fig8_y_d, color=fig8_color_d, labels=all_dict, title=f'Box plot of comparing {fig8_x} and {fig8_y} by {fig8_color}', points='all' if fig8_points else None, log_y=fig8_logy, height=600, width=900)
            fig8.update_traces(marker_line_width=1)
            col16.plotly_chart(fig8)

    
    st.header("Exploring heatmat of 2 num (px.imshow)") # Placeholder - TODO: Change
    col17, col18 = st.columns([2,3])    
    with st.form("Heatmap 2 num"):
        # fig9_x = col17.selectbox("Select a numeric feature for the x-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"]), key=23)
        # fig9_y = col17.selectbox("Select a numeric feature for the y-axis", np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year", fig9_x]), key=24)

        # # Add a correlation check to see if the two numeric features are too correlated
        
        # fig9_x_d = [key for key, value in num_dict.items() if value == fig9_x][0]
        # fig9_y_d = [key for key, value in num_dict.items() if value == fig9_y][0]

        # corr = bmw[[fig9_x_d, fig9_y_d]].corr().iloc[0,1]
        # if abs(corr) > 0.8:
        #     st.warning(f"The correlation between {fig9_x} and {fig9_y} is {corr:.2f}, which is quite high. Consider selecting different features for a more meaningful heatmap.", icon="⚠️")
                
        # usr_sel_bin_num_x = col17.checkbox("Do you want to select the number of bins for x? (Default is 10)", key="usblx9")
        # fig9_bins_x = col17.slider("The amount of bins you want for x", 5, 35, step=1, key="fig9_bins_x") if usr_sel_bin_num_x else 10 

        # usr_sel_bin_num_y = col17.checkbox("Do you want to select the number of bins for y? (Default is 10)", key="usbly9")
        # fig9_bins_y = col17.slider("The amount of bins you want for y", 5, 35, step=1, key="fig9_bins_y") if usr_sel_bin_num_y else 10 

        # submitted = st.form_submit_button("Click to produce the heatmap")
        # if submitted:
        #     heatmap_data = bmw[[fig9_x_d, fig9_y_d]].copy()
        #     heatmap_data['x_binned'] = pd.cut(heatmap_data[fig9_x_d], bins=fig9_bins_x)
        #     heatmap_data['y_binned'] = pd.cut(heatmap_data[fig9_y_d], bins=fig9_bins_y)
        #     # Convert Interval objects to strings for serialization
        #     heatmap_data['x_binned'] = heatmap_data['x_binned'].astype(str)
        #     heatmap_data['y_binned'] = heatmap_data['y_binned'].astype(str)
        #     heatmap_counts = heatmap_data.groupby(['x_binned', 'y_binned']).size().reset_index(name='counts')
        #     heatmap_pivot = heatmap_counts.pivot(index='y_binned', columns='x_binned', values='counts').fillna(0)
        #     fig9 = px.imshow(heatmap_pivot, labels={'x_binned': fig9_x, 'y_binned': fig9_y, 'color':'Count'}, title=f'Heatmap of {fig9_x} and {fig9_y}', aspect='auto', color_continuous_scale="Viridis")
        #     col18.plotly_chart(fig9)

        needed = np.setdiff1d(list(num_dict.values()), ["Sales Count", "Year"])
        fig9_cols = col17.multiselect("Select some numeric features for the heatmap", needed, default=needed, key=25)
        fig9_cols_d = [key for key, value in num_dict.items() if value in fig9_cols]
        submitted = st.form_submit_button("Click to produce the heatmap")
        if submitted:
            if len(fig9_cols_d) < 2:
                st.warning("Please select at least two numeric features to create a heatmap.", icon="⚠️")
            else:
                corr_matrix = bmw[fig9_cols_d].corr()
                fig9 = px.imshow(corr_matrix, text_auto='.4f', aspect='auto', color_continuous_scale='RdBu', title='Correlation Heatmap', labels={'x':'Features', 'y':'Features', 'color':'Correlation'})
                col18.plotly_chart(fig9)


    
    st.header("Chi-sq")
    col19, col20 = st.columns([2,3])
    with st.expander("See explanation on Chi-sq test"):
        st.text("""
                The Chi-squared test is a statistical method used to determine if there is a significant association between two categorical variables. It compares the observed frequencies in each category of a contingency table to the frequencies that would be expected if there were no association between the variables.   
                The test calculates a Chi-squared statistic, which measures the discrepancy between the observed and expected frequencies. A higher Chi-squared value indicates a greater difference between observed and expected frequencies, suggesting a stronger association between the variables.
                The p-value associated with the Chi-squared statistic helps determine the statistical significance of the results. A low p-value (typically less than 0.05) indicates that the observed association is unlikely to have occurred by chance, leading to the rejection of the null hypothesis of independence between the variables.
        """)
    with st.form("Chi-sq"):
        predictor = col19.multiselect("Select some categorical predictors", np.setdiff1d(list(cat_dict.values()), ["Sales Volume Level"]), default=["Car Model", "Color"], key=26)
        target = "Sales Volume Level"
        predictor_d = [key for key, value in cat_dict.items() if value in predictor]
        target_d = [key for key, value in cat_dict.items() if value == target][0]
        logy_scale = col19.checkbox("Click if you want log y scale", key=27)
        submitted = st.form_submit_button("Click to perform Chi-sq test")
        if submitted:
            if len(predictor_d) < 1:
                st.warning("Please select at least one categorical predictor to perform Chi-sq test.", icon="⚠️")
            else:
                chi2_results = []
                for col in predictor_d:
                    contingency_table = pd.crosstab(bmw[col], bmw[target_d])
                    chi2, pval, dof, ex = stats.chi2_contingency(contingency_table)
                    #chi2_fmt = np.where(chi2 < 0.005, f"{chi2:.2e}", f"{chi2:.2f}")
                    #pval_fmt = np.where(pval < 0.0005, f"{pval:.3e}", f"{pval:.2f}")
                    chi2_results.append({
                        'Predictor': all_dict[col],
                        'Chi2 Statistic': chi2,
                        'p-value': pval,
                        'Degrees of Freedom': dof
                    })
                chi2_df = pd.DataFrame(chi2_results).sort_values(by='Chi2 Statistic', ascending=False)
                chi2_df['Chi2 Statistic'] = chi2_df['Chi2 Statistic'].apply(lambda x: np.where(x < 0.005, f"{x:.2e}", f"{x:.2f}"))
                chi2_df['p-value'] = chi2_df['p-value'].apply(lambda x:np.where(x < 0.0005, f"{x:.3e}", f"{x:.2f}"))
                col19.dataframe(chi2_df, use_container_width=True)
                #col19.dataframe(chi2_results, use_container_width=True)
                fig_chi2 = px.bar(chi2_df, x='Predictor', y='Chi2 Statistic', log_y=logy_scale, title='Chi-squared Statistics for Categorical Predictors', labels={'Chi2 Statistic': 'Chi-squared Statistic', 'Predictor': 'Categorical Predictor'}, hover_data={'p-value': ':.4e', 'Degrees of Freedom': True}, text_auto='.1f')
                fig_chi2.update_traces(marker_line_width=1, textposition='outside')
                col20.plotly_chart(fig_chi2)
    


    st.header("Chi-sq 2")
    # col21, col22 = st.columns([2,3])
    with st.form("chi2-simple-form"):
        c1_pretty = st.selectbox("Categorical variable 1", cat_dict.values(), key="chi2_cat1")
        c2_pretty = st.selectbox("Categorical variable 2", np.setdiff1d(list(cat_dict.values()), [c1_pretty]), key="chi2_cat2")
        submitted = st.form_submit_button("Run Chi-square")
        c1 = [k for k, v in cat_dict.items() if v == c1_pretty][0]
        c2 = [k for k, v in cat_dict.items() if v == c2_pretty][0]
        if submitted:        
            # guardrails (columns exist & are categorical-like)
            # if c1 not in bmw.columns or c2 not in bmw.columns:
            #     st.error("One or both selected columns are missing from `bmw`.")
            # else:
            ct = pd.crosstab(bmw[c1], bmw[c2])
            # try:
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            # except ValueError as e:
            #     st.error(f"Chi-square failed: {e}")
            # else:
                # tiny 2-row dataframe with just chi-square and p-value
            # result_df = pd.DataFrame(
            #     {"metric": ["Chi-square", "p-value"],
            #     "value":  [chi2, p]}
            # )
            chi2_fmt = np.where(chi2 < 0.005, f"{chi2:.2e}", f"{chi2:.2f}")
            p_fmt = np.where(p < 0.0005, f"{p:.3e}", f"{p:.2f}")
            result_df = pd.DataFrame({'Chi-square Value':[chi2_fmt], 'P-Value':[p_fmt]})  # TODO: FIgure out how to format these to maybe like .2f or .2e like it depends.
            st.write(f"{c1_pretty} VS {c2_pretty}")
            st.dataframe(result_df)

# ------
if s == 'Naive Bayes Prediction':
    st.title('Naive Bayes Prediction')
    st.text("In this section, we will perform a Naive Bayes classification to predict the 'Sales Volume Level' based on other features in the dataset.")
    with st.expander("See brief info on Naive Bayes Classifier"):
        st.subheader("Custom Naive Bayes Classifier Implementation")
        st.text("The custom Naive Bayes classifier implemented here can handle both categorical and numerical features. It calculates prior probabilities and likelihoods based on the training data, and then makes predictions on the same dataset for demonstration purposes.")
        st.markdown("### Code Implementation")
        st.code("""
    class NullPointerException(ValueError):
        pass
    def custom_naive_bayes(df: pd.DataFrame, target: str) -> pd.DataFrame:
        \"""
        df: DataFrame that already contains the target column
        target: string name of the target column
        Returns: DataFrame with columns [prob_high, prob_low, actual, predicted]
        \"""

        # Assert if DataFrame has NaN values
        assert df.isna().sum(axis=0).sum() == 0, "DataFrame contains NaN values. Please handle them before passing to this function."

        # Step 1
        y = df[target]
        X = df.drop(columns=[target])
        cat_vars = []
        num_vars = []

        cat_vars = X.select_dtypes(include=['object']).columns
        num_vars = X.select_dtypes(include=np.number).columns

        # Step 2 - sorting the high and low dataframes
        df_high = df[df[target] == "High"]
        df_low  = df[df[target] == "Low"]

        # Step 3 - calculating the priors
        prior_high = len(df_high) / len(df)
        prior_low  = len(df_low)  / len(df)
        
        # Step 4
        prob_high, prob_low = {}, {}

        # Step 5
        for col in X.columns:
            if col in cat_vars:
                vals_high = df_high[col].value_counts(normalize=True)
                vals_low  = df_low[col].value_counts(normalize=True)
                V = len(df[col].unique())
                prob_high[col] = df[col].map(vals_high).fillna(1e-8).astype(float)
                prob_low[col]  = df[col].map(vals_low).fillna(1e-8).astype(float)
            elif col in num_vars:
                hi_mean = df_high[col].mean()
                hi_std  = df_high[col].std(ddof=0)
                lo_mean = df_low[col].mean()
                lo_std  = df_low[col].std(ddof=0)
                vals_high = stats.norm(hi_mean, hi_std)
                vals_low  = stats.norm(lo_mean, lo_std)
                prob_high[col] = vals_high.pdf(df[col])
                prob_low[col]  = vals_low.pdf(df[col])
            else:
                raise NullPointerException(f"Column '{col}' is neither categorical nor numeric. An error occured.")

        df_high = pd.DataFrame(prob_high).prod(axis=1) * prior_high
        df_low  = pd.DataFrame(prob_low).prod(axis=1) * prior_low

        preds = np.where(df_high > df_low, "High", "Low")

        # Step 6

        out = pd.DataFrame({
            "prob_high": df_high.map(lambda v: f'{v:2e}'),
            "prob_low": df_low.map(lambda v: f'{v:2e}'),
            "actual": y,
            "predicted": preds,
        })

        accuracy = np.mean(out["actual"] == out["predicted"])

        return out, accuracy
        """, language='python')
        st.text("The above code defines a custom Naive Bayes classifier that processes both categorical and numerical features. It calculates the prior probabilities and likelihoods, then makes predictions based on these calculations.")
    # def facet_box_numeric_vs_target(df:pd.DataFrame, numeric_cols:list, target:str='Sales Classification', wrap:int=3, bins_if_numeric_target:int=4, column_to_put_chart=None):
    # # If the target is numeric, bin it so box plots make sense
    #     if np.issubdtype(df[target].dtype, np.number):
    #         labels = [f"Q{i+1}" for i in range(bins_if_numeric_target)]
    #         try:
    #             df["_target_cat_"] = pd.qcut(df[target], q=bins_if_numeric_target, labels=labels, duplicates="drop")
    #         except Exception:
    #             df["_target_cat_"] = pd.cut(df[target], bins=bins_if_numeric_target, labels=labels)
    #         target_cat = "_target_cat_"
    #     else:
    #         target_cat = target
    #     if not numeric_cols:
    #         st.info("No numeric predictors to facet.")
    #         return

    #     melted = df[numeric_cols + [target_cat]].melt(id_vars=target_cat, var_name="feature", value_name="value")
    #     melted = melted.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])

    #     n_panels = melted["feature"].nunique()
    #     rows = math.ceil(n_panels / wrap)

    #     fig = px.box(
    #         melted,
    #         x=target_cat, y="value",
    #         facet_col="feature", facet_col_wrap=wrap,  # ← THIS is the facet
    #         points="outliers",
    #         title="Facet box plots: numeric predictors vs target"
    #     )
    #     fig.update_layout(height=max(520, rows*320), margin=dict(t=60, b=30))
    #     if column_to_put_chart:
    #         column_to_put_chart.plotly_chart(fig, use_container_width=True)
    #     else:
    #         st.plotly_chart(fig, use_container_width=True)
    # col_nb_1, col_nb_2 = st.columns([2,3])

    # def fit_distribution(data):
    #     mu = np.mean(data)
    #     sigma = np.std(data)
    #     return stats.norm(mu, sigma)

    # def sales_predict(df, target_col='sales_classification'):
    #     # expects labels 'High' and 'Low'
    #     high = df[df[target_col] == 'High'].copy()
    #     low  = df[df[target_col] == 'Low'].copy()

    #     priors = df[target_col].value_counts(normalize=True)
    #     h_prior = priors.get('High', 0.0)
    #     l_prior = priors.get('Low',  0.0)

    #     p_h, p_l = {}, {}
    #     for col in df.columns:
    #         if col == target_col:
    #             continue

    #         # categorical: normalized value_counts (no Laplace)
    #         if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
    #             h_con = dict(high[col].value_counts(normalize=True))
    #             l_con = dict(low[col].value_counts(normalize=True))
    #             p_h[col] = df[col].map(h_con)
    #             p_l[col] = df[col].map(l_con)

    #         # numeric: Gaussian pdf (Harry’s approach)
    #         elif str(df[col].dtype) == 'float64':
    #             p_h[col] = fit_distribution(high[col]).pdf(df[col])
    #             p_l[col] = fit_distribution(low[col]).pdf(df[col])

    #         # (optional) treat ints as numeric too:
    #         # elif np.issubdtype(df[col].dtype, np.number):
    #         #     p_h[col] = fit_distribution(high[col].astype(float)).pdf(df[col].astype(float))
    #         #     p_l[col] = fit_distribution(low[col].astype(float)).pdf(df[col].astype(float))

    #     final_h = pd.DataFrame(p_h).prod(axis=1) * h_prior
    #     final_l = pd.DataFrame(p_l).prod(axis=1) * l_prior

    #     predicted = (final_h > final_l).map({True: 'High', False: 'Low'})
    #     final = pd.concat([predicted, df[target_col]], axis=1)
    #     final.columns = ['Predicted', 'Actual']
    #     acc = (final['Predicted'] == final['Actual']).mean()

    #     out = pd.concat(
    #         [final,
    #         final_h.rename('P(High)'),
    #         final_l.rename('P(Low)')],
    #         axis=1
    #     )
    #     return out, float(acc)
    with st.expander("See information on the Naive Bayes Classifier:"):
        st.text("""
The Naive Bayes classifier is a simple but powerful method used to make predictions based on probabilities. It is most commonly applied in text classification tasks, such as detecting spam emails or identifying the sentiment of a review. The central idea is to use evidence, or features, to estimate how likely something belongs to a particular category.

The “Bayes” part of the name comes from Bayes' Theorem, a rule in probability that allows us to update our beliefs when we see new evidence. In a spam detection example, the model asks: if an email contains the word “free,” how likely is it that the email is spam? It combines the likelihood of each word with the overall chance that any email is spam, producing a final probability for whether a specific email is spam or not.

The “naive” part refers to an important assumption the model makes: that every piece of evidence, or feature, is independent of the others. In other words, it assumes that the presence of one word in an email has nothing to do with another word. In reality, this is rarely true—words like “winner” and “prize” often appear together—but even with this oversimplified assumption, the method tends to perform surprisingly well.

Because of its simplicity, Naive Bayes is extremely fast to train and works well even with relatively small datasets. It often serves as a reliable baseline model before using more complex algorithms. In essence, Naive Bayes acts like a very straightforward reasoning process: it looks at each clue, measures how strongly it supports one category over another, and combines those pieces of evidence to make a final, probabilistic decision.
                """)
    st.header("Accuracy test")
    with st.form("Accuracy"):
        # --- multiselects show PRETTY names ---
        # exclude the target's pretty name ("Sales Volume Level") from categorical options
        all_pretty_opts = np.setdiff1d(list(all_dict.values()), ['Sales Volume Level'])
        all_ms = st.multiselect("Predictors", all_pretty_opts, key=4001)

        # PRETTY -> DIRTY using your list-comprehension style
        preds = [k for k, v in all_dict.items() if v in all_ms]

        submitted = st.form_submit_button("Submit to view predictions and accuracy")

        if submitted:
            output, accuracy = custom_naive_bayes(bmw[preds + ['sales_classification']], target='sales_classification')
            output_clean = output.copy()
            # TODO: Change the prob_high and prob_low column values to .2e or .3e (depends) and do it in one line of code using .map and the lambda function.
            # TODO: Make the column names to more user-friendly names so that they are more nicer like the prediction and actual. for probs dont do abbreviations
            st.dataframe(output, use_container_width=True)
            st.write(f"The accuracy of this prediction is: {accuracy:.3f}")
            # if len(predictors) == 0:
            #     st.warning("Select at least one predictor.")
            # elif 'sales_classification' not in bmw.columns:
            #     st.error("Column 'sales_classification' not found in `bmw`.")
            # else:
            #     use_cols = predictors + ['sales_classification']
            #     df_in = bmw[use_cols].dropna().copy()

            #     predicted, acc = sales_predict(df_in, target_col='sales_classification')
            #     st.markdown(f"**Accuracy:** `{acc:.4f}`")
            #     st.dataframe(predicted)

            #     # ---------- Chi-square (categorical only) ----------
            #     st.subheader("Chi-square (categorical predictors vs sales_classification)")
            #     cat_X = [c for c in predictors if str(bmw[c].dtype) in ('object','category')]
            #     if cat_X:
            #         rows = []
            #         for col in cat_X:
            #             ct = pd.crosstab(bmw[col], bmw['sales_classification'])
            #             #try:
            #             chi2, p, dof, _ = stats.chi2_contingency(ct)
            #             # show pretty name if available
            #             pretty_name = cat_dict.get(col, col)
            #             rows.append({"Predictor": pretty_name, "Chi2": chi2, "dof": dof, "p-value": p})
            #             # except ValueError:
            #             #     rows.append({"Predictor": cat_dict.get(col, col), "Chi2": np.nan, "dof": np.nan, "p-value": np.nan})
            #         st.dataframe(pd.DataFrame(rows).sort_values("p-value", na_position="last"))
            #     else:
            #         st.info("No categorical predictors selected.")

            #     # ---------- Plotly Express facet box (numeric vs target) ----------
            #     st.subheader("Facet box plot (numeric predictors vs sales_classification)")
            #     num_X = [c for c in predictors if c not in cat_X]
            #     if num_X:
            #         melt_cols = num_X + ['sales_classification']
            #         long_df = bmw[melt_cols].dropna().melt(
            #             id_vars='sales_classification',
            #             value_vars=num_X,
            #             var_name='predictor_dirty',
            #             value_name='Value'
            #         )
            #         # replace dirty with PRETTY in the facet labels
            #         pretty_map_num = {k: v for k, v in num_dict.items() if k in num_X}
            #         long_df['Predictor'] = long_df['predictor_dirty'].map(lambda k: pretty_map_num.get(k, k))

            #         fig = px.box(
            #             long_df,
            #             x='sales_classification',
            #             y='Value',
            #             facet_col='Predictor',
            #             facet_col_wrap=3,
            #             points='outliers',
            #             title='Numeric predictors by sales_classification'
            #         )
            #         fig.update_layout(height=350 * int(np.ceil(len(long_df['Predictor'].unique()) / 3)))
            #         st.plotly_chart(fig, use_container_width=True)
            #     else:
            #         st.info("No numeric predictors selected.")

    with st.expander("View the function"):
        # st.code(fit_distribution, language='python')
        # st.code(profit_predict, language='python')
        st.text("No actual information yet. Still under construction. Come back later for more information.")

if s == "Analysis":
    st.header("The analysis Section")

    # Here I will write the overall purpose for this analysis section as well as "the general overview of how these results will guide me in feature selection for the NB Classifier"
    st.text("These analysis plots (Chi-square tests and heatmaps) can let me know about the correlation between the different columns and know what features to and not to select (to avoid some being over-correlated) for my feature selection.")

    mileage_feats = pd.Series(["mileage_km", "mileage_per_year", "price_per_100km"]).replace(all_dict).apply(lambda text:"<b>"+text+"</b>")
    st.subheader("Facet")
    bmw_melted = bmw.melt(
        id_vars='sales_classification',
        value_vars=list(num_dict.keys()),
        var_name='Features'
    )
    bmw_melted['Features'] = bmw_melted['Features'].replace(all_dict).apply(lambda text:"<b>"+text+"</b>")
    figure_mf = px.box(bmw_melted[bmw_melted['Features'].isin(mileage_feats)], 'sales_classification', 'value', facet_col='Features', title='Mileage features by Sales Volume Level', color='sales_classification', height=600, facet_col_spacing=0.05, labels=all_dict, log_y=True)
    figure_mf.update_yaxes(dtick=1)
    figure_mf.for_each_annotation(lambda x:x.update(text=x.text.split("=")[-1]))
    st.plotly_chart(figure_mf, use_container_width=True)

    figure_nu = px.box(bmw_melted[~bmw_melted['Features'].isin(mileage_feats)], 'sales_classification', 'value', facet_col='Features', title='Other Numeric features by Sales Volume Level', color='sales_classification', height=600, facet_col_spacing=0.06, labels=all_dict, facet_col_wrap=3, facet_row_spacing=0.09)
    figure_nu.update_yaxes(showticklabels=True, matches=None)
    figure_nu.for_each_annotation(lambda x:x.update(text=f"<b>{x.text.split('=')[-1]}</b>", font=dict(size=14, color="black", family="Arial")))
    st.plotly_chart(figure_nu)


    st.subheader("Chi-sq") # TODO: Add all columns to the chi-sqm similar to the "chisq" exploratory graph but you choose all
    # with st.form("Chi-sq"):
    #         predictor = col19.multiselect("Select some categorical predictors", np.setdiff1d(list(cat_dict.values()), ["Sales Volume Level"]), default=["Car Model", "Color"], key=26)
    #         target = "Sales Volume Level"
    #         predictor_d = [key for key, value in cat_dict.items() if value in predictor]
    #         target_d = [key for key, value in cat_dict.items() if value == target][0]
    #         logy_scale = col19.checkbox("Click if you want log y scale", key=27)
    #         submitted = st.form_submit_button("Click to perform Chi-sq test")
    #         if submitted:
    #             if len(predictor_d) < 1:
    #                 st.warning("Please select at least one categorical predictor to perform Chi-sq test.", icon="⚠️")
    #             else:
    #                 chi2_results = []
    #                 for col in predictor_d:
    #                     contingency_table = pd.crosstab(bmw[col], bmw[target_d])
    #                     chi2, pval, dof, ex = stats.chi2_contingency(contingency_table)
    #                     #chi2_fmt = np.where(chi2 < 0.005, f"{chi2:.2e}", f"{chi2:.2f}")
    #                     #pval_fmt = np.where(pval < 0.0005, f"{pval:.3e}", f"{pval:.2f}")
    #                     chi2_results.append({
    #                         'Predictor': all_dict[col],
    #                         'Chi2 Statistic': chi2,
    #                         'p-value': pval,
    #                         'Degrees of Freedom': dof
    #                     })
    #                 chi2_df = pd.DataFrame(chi2_results).sort_values(by='Chi2 Statistic', ascending=False)
    #                 chi2_df['Chi2 Statistic'] = chi2_df['Chi2 Statistic'].apply(lambda x: np.where(x < 0.005, f"{x:.2e}", f"{x:.2f}"))
    #                 chi2_df['p-value'] = chi2_df['p-value'].apply(lambda x:np.where(x < 0.0005, f"{x:.3e}", f"{x:.2f}"))
    #                 col19.dataframe(chi2_df, use_container_width=True)
    #                 #col19.dataframe(chi2_results, use_container_width=True)
    #                 fig_chi2 = px.bar(chi2_df, x='Predictor', y='Chi2 Statistic', log_y=logy_scale, title='Chi-squared Statistics for Categorical Predictors', labels={'Chi2 Statistic': 'Chi-squared Statistic', 'Predictor': 'Categorical Predictor'}, hover_data={'p-value': ':.4e', 'Degrees of Freedom': True}, text_auto='.1f')
    #                 fig_chi2.update_traces(marker_line_width=1, textposition='outside')
    #                 col20.plotly_chart(fig_chi2)
    tgt = "sales_classification"
    preds = list(cat_dict.keys())
    preds = np.setdiff1d(preds, [tgt])
    chi2_re = []
    for col in preds:
        contingency_table = pd.crosstab(bmw[col], bmw[tgt])
        chi2, pval, dof, ex = stats.chi2_contingency(contingency_table)
        chi2_re.append({
            'Predictor': all_dict[col],
            'Chi2 Stats': chi2,
            'p_value': pval,
            "Degrees of Freedom": dof,
        })
    chi2_df = pd.DataFrame(chi2_re).sort_values(by='Chi2 Stats', ascending=False)
    chi2_df['Chi2 Stats'] = chi2_df['Chi2 Stats'].apply(lambda x: np.where(x < 0.005, f"{x:.2e}", f"{x:.2f}"))
    chi2_df['p_value'] = chi2_df['p_value'].apply(lambda x:np.where(x < 0.0005, f"{x:.3e}", f"{x:.2f}"))
    st.dataframe(chi2_df, use_container_width=True)
    fig_chi2_a = px.bar(chi2_df, x='Predictor', y='Chi2 Stats', title='', labels={'Chi2 Statistic': 'Chi-squared Statistic', 'Predictor': 'Categorical Predictor'}, hover_data={'p_value': ':.4e', 'Degrees of Freedom': True}, text_auto='.1f')
    st.plotly_chart(fig_chi2_a)


    st.subheader("Heatmap of numeric cols corr.") # TODO: Use px.imshow()
    st.text("This heatmap shows the correlation between the numeric columns so that it prepares us for the later feature selection.")
    corr_matrix = bmw[np.setdiff1d(list(num_dict.keys()), ['year'])].corr()
    heatmap = px.imshow(corr_matrix, text_auto='.4f', aspect='auto', color_continuous_scale='RdBu', title='Correlation Heatmap', labels={'x':'Features', 'y':'Features', 'color':'Correlation'})
    st.plotly_chart(heatmap)

    # Show all the steps (each collection of features) and the reason for choosing that particular set of features and the results. EG: Include one of using all the features because none of them are highly correlated but there are also no individually strong predictors.
