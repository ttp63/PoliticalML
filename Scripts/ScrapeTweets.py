# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
# Import libraries and setup for scrape

import GetOldTweets3 as got
import pandas as pd
import os

# Scraping Parameters
start_date = "2020-04-23"
end_date = "2020-04-26"
max_tweet = 10000
group = "sen"  # 'sen' or 'rep' (for now)
obs = [
    "to",
    "id",
    "hashtags",
    "mentions",
    "geo",
]  # Parameter for column formatting used later on

# Check for existing csv file
data_file_list = os.listdir("../Data")
filename = group + ".csv"

if (filename) in data_file_list:
    base_df = pd.read_csv("../Data/" + filename, parse_dates=["date"])
else:
    base_df = pd.read_csv("../Data/test.csv", parse_dates=["date"])
    base_df = base_df[0:0]

# Correct column formats
base_df[obs] = base_df[obs].astype(object)

# Initialize df to export later
new_df = base_df

# CSV pulled from https://github.com/unitedstates/congress-legislators (lightly modified to correct some twitter accounts)
congress = pd.read_csv("../Data/legislators-current.csv")
congress_sub = congress[congress["type"] == group]

#%%
# Get the scrape on
# Can scrape about 1 month of data (~10000 tweets) before getting kicked out by twitter, switch vpn network between runs
for index, row in congress_sub.iterrows():
    username = row["twitter"]
    print("Scraping twitter account " + username + "...")

    tweetCriteria = (
        got.manager.TweetCriteria()
        .setUsername(username)
        .setSince(start_date)
        .setUntil(end_date)
        .setMaxTweets(max_tweet)
    )

    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    tweets_df = pd.DataFrame.from_records([t.__dict__ for t in tweets])
    new_df = new_df.append(tweets_df, ignore_index=True)

#%%
# Prepare and export to csv
# Delete duplicates
new_df[["id"]] = new_df[["id"]].apply(pd.to_numeric)
return_df = new_df.drop_duplicates(subset="id")

# Export
return_df.to_csv(path_or_buf="../Data/" + filename, index=False)


# %%
# Doublechecks before export (if wanted)
counts_scraped = new_df.groupby(["username"]).size().to_frame()
counts_scraped["twitter"] = counts_scraped.index
counts_scraped.columns = ["count", "twitter"]
counts = (
    congress_sub["twitter"].to_frame().merge(counts_scraped, how="left", on="twitter")
)
unique_check = counts.twitter.value_counts()

# %%


# %%
