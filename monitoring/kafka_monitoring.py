import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__name__)))
from kafka import KafkaConsumer
from prometheus_client import Counter, Histogram, start_http_server
import db_utils.config as config
import db_utils.utils as utils

start_http_server(8765)

# Metrics like Counter, Gauge, Histogram, Summaries
# Refer https://prometheus.io/docs/concepts/metric_types/ for details of each metric
REQUEST_COUNT = Counter(
    'request_count', 'Recommendation Request Count',
    ['http_status']
    )

REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 
    'Request latency'
    )


EVEN_RATING_COUNT = Histogram(
    'even_rating',
    'Even Rating',
    buckets=(1, 2, 3, 4, 5)
)

ODD_RATING_COUNT = Histogram(
    'odd_rating',
    'Odd Rating',
    buckets=(1, 2, 3, 4, 5)
)

HIT_RATE = Counter(
    'hit_rate', 'Hit rate for models',
    ['endpoint']
)

def main():
    consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        group_id="prometest", # New group for monitoring
        enable_auto_commit=True,
        auto_commit_interval_ms=1000
    )

    # userid: most recent recommendations
    recommendations = {}

    for message in consumer:
        event = message.value.decode('utf-8')
        kind, parsed_data = utils.parse_line(event)
        if kind == 'recommendation':
            REQUEST_COUNT.labels(http_status=parsed_data['status']).inc()
            REQUEST_LATENCY.observe(parsed_data['responsetime']/1000)
            if parsed_data['status'] == 200:
                parsed_data['results'].append(parsed_data['server'])
                recommendations[parsed_data['userid']] = parsed_data['results']
        elif kind == 'rate':
            if parsed_data['userid'] % 2 == 0:
                EVEN_RATING_COUNT.observe(parsed_data['rating'])
            else:
                ODD_RATING_COUNT.observe(parsed_data['rating'])
        elif kind == 'data':
            results = recommendations[parsed_data['userid']] if parsed_data['userid'] in recommendations else {}
            if parsed_data['movieid'] in results:
                print("hit")
                HIT_RATE.labels(endpoint=results[-1]).inc()
                recommendations[parsed_data['userid']].remove(parsed_data['movieid'])
        else:
            # malformated
            pass

if __name__ == "__main__":
    main()
