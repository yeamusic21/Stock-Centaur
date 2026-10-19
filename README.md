<center><h1>Stock-Centaur</h1></center>
<center><p>Web tool for evaluating stocks and portfolios leveraging AI and ML</p></center>
<center><img src="stock_centaur_logo.jpeg" width="300" height="300"></center>


##### 
# Road Map

  - [x] Slick web design with side panel listing options
  - [x] Responsive and aesthetically pleasing table
  - [x] Stocks sorted by 'trend' varible (combo of 5yr, 1yr, & 1mon slope)
  - [ ] Portfolio sandbox where users can select stocks and create a hypothetical portfolio with projected earnings
    - Powered by a tree-based model that predicts each stock's price 30 days out from trend/slope features (1yr, 6mon, 3mon, 1mon, 2wk, 3day), used to run portfolio simulations
  - [ ] Individual stock overview, similar to Yahoo Finance, but with a focus on ML & AI insights
    - LLM-powered research is an optional setting, enabled only when the user adds their own OpenAI API key

# Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. pull 5 years of daily prices for every ticker (~15 min; use --limit 200 for a quick test)
python src/modeling/data_gathering/01_pull_stock_data.py

# 2. compute slope features and the trend rankings
python src/modeling/02_build_trend_table.py

# 3. run the web app at http://127.0.0.1:5000
python src/app/app.py
```

You can also rebuild the data from the app with the **Refresh data** button on the Trend Rankings page. Data is written to `data/` (git-ignored). See the Methodology page in the app for how the trend score is calculated.
