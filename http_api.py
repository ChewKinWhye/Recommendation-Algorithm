from content_filtering_recommendation import content_query_API
from collaborative_filtering_recommendation import collaborative_query_API
from flask import Flask, json, request


def get_recommendations(json_file):
    features = ["1"]
    features.extend(json.load(json_file))
    content_recommendations = content_query_API(features)
    collaborative_recommendations, taste_breaker_recommendations = collaborative_query_API(features)
    total_recommendations = content_recommendations
    total_recommendations.extend(collaborative_recommendations)
    total_recommendations.extend(taste_breaker_recommendations)
    return json.dumps(total_recommendations)


companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

api = Flask(__name__)


@api.route('/json-example', methods=['POST']) #GET requests will be blocked
def get_companies():
    req_data = request.get_json()
    recommendations = get_recommendations(req_data)
    return recommendations


if __name__ == '__main__':
    api.run()

