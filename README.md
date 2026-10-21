<center><h1>Stock-Centaur</h1></center>
<center><p>Web tool for evaluating stocks and portfolios leveraging AI and ML</p></center>
<center><img src="stock_centaur_logo.jpeg" width="300" height="300"></center>


##### 
# Road Map

  - [x] Slick web design with side panel listing options
  - [x] Responsive and aesthetically pleasing table
  - [x] Stocks sorted by 'trend' varible (combo of 5yr, 1yr, & 1mon slope)
  - [x] Portfolio sandbox where users can select stocks and create a hypothetical portfolio with projected earnings
    - Powered by a tree-based model that predicts each stock's price 30 days out from trend, risk, beta, industry-momentum and market-regime features, used to run portfolio simulations
    - The Monte Carlo runs 10,000 correlated simulations of the model's forecasts, for any ticker (ETFs and other unranked tickers are scored on demand)
    - Save sandbox portfolios by title (stored locally with TinyDB, in `data/stock_centaur.json`)
  - [x] Individual stock overview, similar to Yahoo Finance, but with a focus on ML & AI insights
    - Price trend with the model's 30-day forecast range, trend score, risk stats, company profile and recent news
    - LLM-powered research runs on a free local model in [LM Studio](https://lmstudio.ai), no API key needed
  - [x] Connections page: pull Robinhood and E*Trade holdings into saved portfolios (you sign in on each pull; logins are never stored) and pick the LM Studio model
  - [x] Research agent: pick a portfolio and it pulls fresh prices, scrapes news from free RSS feeds and article pages, runs the model and Monte Carlo, and has the local LLM write a review

# Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. pull 5 years of daily prices and volumes for every ticker (~15 min; use --limit 200 for a quick test)
python src/modeling/data_gathering/01_pull_stock_data.py

# 1b. pull sector, industry and earnings dates (~1.5 hours the first time, cached after that)
python src/modeling/data_gathering/01b_pull_company_data.py

# 2. compute slope features and the trend rankings
python src/modeling/02_build_trend_table.py

# 3. train the 30-day price model (~10 min; prints walk-forward validation metrics vs a no-change baseline)
python src/modeling/03_train_price_model.py

# 4. predict each ranked stock's price 30 days out for the Portfolio Sandbox
python src/modeling/04_predict_prices.py

# 5. run the web app at http://127.0.0.1:5000
python src/app/app.py
```

### Connections and the research agent

- **LM Studio**: install [LM Studio](https://lmstudio.ai), download a chat model, and start its local server (Developer tab, default `http://localhost:1234`). The Connections page can launch LM Studio for you if you give it the start command (e.g. `~/bin/lmstudio`). Small models on a CPU can take a few minutes per report.
- **Robinhood**: enter your username and password (and an authenticator code if you use one) on the Connections page. Robinhood usually asks you to approve the sign-in in its app, or texts a code you enter on the page. This uses the unofficial API through `robin_stocks`.
- **E*Trade**: E*Trade's API doesn't accept a username and password directly. Create an API key at [developer.etrade.com](https://developer.etrade.com/getting-started), enter the consumer key and secret, sign in on E*Trade's own page with your username and password, and paste back the verification code.

Logins are used for that one pull and never saved; only the holdings are stored, in `data/stock_centaur.json`.

You can also rebuild the data, retrain the model and refresh predictions from the app with the **Refresh data** button on the Trend Rankings page. Data is written to `data/` (git-ignored). See the Methodology page in the app for how the trend score is calculated.
