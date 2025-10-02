import requests
import pandas as pd

API_KEY = "8d5eddc324a617e770800433fa387fe7"
CITY = "New York"

url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"

r = requests.get(url).json()

data = []
for x in r["list"]:
    temp = x["main"]["temp"]
    hum = x["main"]["humidity"]
    rain = x.get("rain", {}).get("3h", 0)
    data.append([temp, hum, rain])

df = pd.DataFrame(data, columns=["temp","humidity","rainfall"])
df.to_csv("../data/weather.csv", index=False)
print("saved weather.csv with", len(df), "rows")
