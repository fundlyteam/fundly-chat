import csv
import json
from datetime import datetime
from google_play_scraper import app, Sort, reviews

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return super().default(obj)

def save_reviews_to_csv(reviews_data):
    with open("reviews.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(reviews_data.keys()))
        writer.writerow(reviews_data)

def read_app_list(file_path):
    app_list = []
    with open(file_path, "r") as file:
        for line in file:
            app_id = line.strip()
            if app_id:
                app_list.append(app_id)
    return app_list


with open("reviews.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Review ID", "User Name", "User Image", "Content", "Score", "Thumbs Up Count", "Review Created Version", "At", "Reply Content", "Replied At", "App Version", "App ID"])
    writer.writeheader()


if __name__ == "__main__":
    app_list = read_app_list("app_list.txt")
    for app_id in app_list:
        print(app_id)
        result, continuation_token = reviews(
            app_id,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=3
        )
        print(f"Printed {len(result)} reviews for app {app_id}")
        for review in result:
            review["App ID"] = app_id
            save_reviews_to_csv(review)
