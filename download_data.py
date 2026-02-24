from sec_edgar_downloader import Downloader

# email reuqired here to download the reports, you can use any email address
dl = Downloader("Adil_Student", "muhammadaadilusmani@gmail.com")

# Tesla (TSLA) ki latest 10-K report (2023-2024)
print("Downloading Tesla Report...")
dl.get("10-K", "TSLA", after="2024-01-01")

# Apple (AAPL) ki latest 10-K report
print("Downloading Apple Report...")
dl.get("10-K", "AAPL", after="2024-01-01")

print("Download Complete! Check 'sec-edgar-filings' folder.")