import yfinance as yf

def fetch_data():
    df=yf.download('^NSEI',
                   interval="5m",
                   period="60d",
                   progress=False)
    df.to_csv("ml/data/nifty_5m.csv")


if __name__ == "__main__":
    fetch_data()