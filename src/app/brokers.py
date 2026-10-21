"""Pull holdings from Robinhood and E*Trade into saved portfolios.

Nothing is remembered between pulls: you sign in each time, the holdings are saved, and the session
is dropped.

Robinhood: signs in with your username and password (plus an authenticator code if you use one)
through robin_stocks, which uses Robinhood's unofficial API. Robinhood usually asks you to approve
the sign-in in its app or with a code it texts/emails you; the pull waits for that, and a code can
be sent from the Connections page while it waits.

E*Trade: its API doesn't accept a username and password. You create an API key (consumer key and
secret) at developer.etrade.com, the app asks E*Trade for a sign-in link, you sign in on E*Trade's
own site with your username and password, and paste back the verification code it shows.
"""
import threading
from datetime import datetime

import store

ROBINHOOD_APPROVAL_TIMEOUT = 300  # seconds to wait for a texted/emailed code

_lock = threading.Lock()
_status = {"robinhood": {"state": "idle"}, "etrade": {"state": "idle"}}
_code = {"event": threading.Event(), "value": None}
_etrade = {}  # OAuth handshake in progress: {"oauth", "dev"}


def status():
    with _lock:
        last = {s: None for s in ("robinhood", "etrade")}
        for p in store.list_portfolios():
            if p["source"] in last and (last[p["source"]] is None or p["updated_at"] > last[p["source"]]["updated_at"]):
                last[p["source"]] = p
        return {k: {**v, "last": last[k]} for k, v in _status.items()}


def _update(broker, **fields):
    with _lock:
        _status[broker].update(fields)


def _finish(broker, saved):
    _update(broker, state="done", message=f"Saved {len(saved)} portfolio{'s' if len(saved) != 1 else ''}.",
            saved=saved, finished_at=datetime.now().isoformat(timespec="seconds"))


def _fail(broker, message):
    _update(broker, state="error", message=message, finished_at=datetime.now().isoformat(timespec="seconds"))


# ---------- Robinhood ----------
def _wait_for_code(prompt):
    """robin_stocks calls input() when Robinhood texts or emails a code; ask the web page for it instead."""
    _code["event"].clear()
    _update("robinhood", state="needs_code", message=prompt.strip().rstrip(":") + ".")
    if not _code["event"].wait(ROBINHOOD_APPROVAL_TIMEOUT):
        raise TimeoutError("Timed out waiting for the verification code.")
    _update("robinhood", state="running", message="Checking the code…")
    return _code["value"]


def submit_robinhood_code(code):
    with _lock:
        if _status["robinhood"].get("state") != "needs_code":
            return False
    _code["value"] = str(code).strip()
    _code["event"].set()
    return True


def _robinhood_pull(username, password, mfa_code):
    import robin_stocks.robinhood as rh
    from robin_stocks.robinhood import authentication

    authentication.input = _wait_for_code
    try:
        _update("robinhood", message="Signing in… if Robinhood sends an approval request to your phone, approve it.")
        # its own (deleted) token file name so this never touches a session another robin_stocks script saved
        rh.login(username, password, mfa_code=mfa_code or None, store_session=False, expiresIn=3600,
                 pickle_name="_stock_centaur")
        if not rh.helper.LOGGED_IN:
            raise RuntimeError("Robinhood didn't accept the sign-in. Check your username, password and code.")
        _update("robinhood", message="Downloading holdings…")
        holdings = rh.account.build_holdings()
        account = (rh.profiles.load_account_profile(info="account_number") or "default")
        rows = [{"ticker": t, "mode": "shares", "value": float(h["quantity"])}
                for t, h in holdings.items() if float(h.get("quantity") or 0) > 0]
        if not rows:
            raise RuntimeError("Signed in, but the account has no stock holdings.")
        cost = {t: float(h.get("average_buy_price") or 0) for t, h in holdings.items()}
        pid = store.save_portfolio(f"Robinhood ••{str(account)[-4:]}", rows, source="robinhood", account=str(account),
                                   extra={"cost_basis": cost})
        _finish("robinhood", [pid])
    except Exception as err:
        _fail("robinhood", str(err) or err.__class__.__name__)
    finally:
        authentication.input = input
        try:
            rh.logout()
        except Exception:
            pass


def start_robinhood(username, password, mfa_code=None):
    if not username or not password:
        raise ValueError("Enter your Robinhood username and password.")
    with _lock:
        if _status["robinhood"].get("state") in ("running", "needs_code"):
            return False
        _status["robinhood"] = {"state": "running", "message": "Starting…",
                                "started_at": datetime.now().isoformat(timespec="seconds")}
    threading.Thread(target=_robinhood_pull, args=(username, password, mfa_code), daemon=True).start()
    return True


# ---------- E*Trade ----------
def etrade_authorize_url(consumer_key, consumer_secret, sandbox=False):
    """Step 1: get the E*Trade page where you sign in and receive a verification code."""
    from pyetrade import ETradeOAuth

    if not consumer_key or not consumer_secret:
        raise ValueError("Enter your E*Trade consumer key and secret.")
    oauth = ETradeOAuth(consumer_key.strip(), consumer_secret.strip())
    try:
        url = oauth.get_request_token()
    except Exception as err:
        raise ValueError(f"E*Trade rejected the API key: {err}") from err
    with _lock:
        _etrade.clear()
        _etrade.update(oauth=oauth, dev=bool(sandbox), keys=(consumer_key.strip(), consumer_secret.strip()))
        _status["etrade"] = {"state": "awaiting_code", "message": "Sign in on E*Trade, then paste the verification code."}
    return url


def _etrade_positions(accounts, account_key):
    rows, page = [], 1
    while True:
        res = accounts.get_account_portfolio(account_key, resp_format="json", page_number=page)
        portfolios = ((res or {}).get("PortfolioResponse") or {}).get("AccountPortfolio") or []
        for p in portfolios:
            for pos in p.get("Position", []):
                product = pos.get("Product", {})
                if product.get("securityType") in ("EQ", "MF") and float(pos.get("quantity") or 0) > 0:
                    rows.append({"ticker": product.get("symbol") or pos.get("symbolDescription"),
                                 "mode": "shares", "value": float(pos["quantity"]),
                                 "cost": float(pos.get("pricePaid") or 0)})
        total_pages = max([int(p.get("totalPages", 1)) for p in portfolios] or [1])
        if page >= total_pages:
            return rows
        page += 1


def etrade_pull(verifier):
    """Step 2: trade the verification code for a session and save every brokerage account's holdings."""
    from pyetrade import ETradeAccounts

    with _lock:
        handshake = dict(_etrade)
    if not handshake:
        raise ValueError("Start by getting a sign-in link.")
    _update("etrade", state="running", message="Downloading accounts…")
    try:
        tokens = handshake["oauth"].get_access_token(str(verifier).strip())
        accounts = ETradeAccounts(*handshake["keys"], tokens["oauth_token"], tokens["oauth_token_secret"],
                                  dev=handshake["dev"])
        listing = accounts.list_accounts(resp_format="json")
        account_list = listing["AccountListResponse"]["Accounts"]["Account"]
        saved = []
        for acct in account_list:
            if acct.get("accountStatus") == "CLOSED":
                continue
            positions = _etrade_positions(accounts, acct["accountIdKey"])
            if not positions:
                continue
            name = acct.get("accountDesc") or acct.get("accountName") or "Account"
            cost = {p["ticker"]: p.pop("cost") for p in positions}
            saved.append(store.save_portfolio(f"E*Trade {name} ••{str(acct['accountId'])[-4:]}", positions,
                                              source="etrade", account=str(acct["accountId"]),
                                              extra={"cost_basis": cost}))
        if not saved:
            raise RuntimeError("Signed in, but none of the accounts hold stocks or funds.")
        _finish("etrade", saved)
    except Exception as err:
        _fail("etrade", str(err) or err.__class__.__name__)
        raise ValueError(str(err) or err.__class__.__name__) from err
    finally:
        with _lock:
            _etrade.clear()
    return status()["etrade"]
