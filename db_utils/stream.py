import config
import utils
from kafka import KafkaConsumer
import psycopg2
import argparse
import time

def database_insert(buffer):
    view_lines = []
    rate_lines = []
    recommendation_lines = []
    error_lines = []

    for line in buffer:
        kind, parsed_data = utils.parse_line(line)
        if kind == 'data':
            view_lines.append(parsed_data)
        elif kind == 'rate':
            rate_lines.append(parsed_data)
        elif kind == 'recommendation':
            recommendation_lines.append(parsed_data)
        else:
            error_lines.append({"type": "parse", "line": line})
            

    conn = utils.get_connection()

    # Insert other info
    utils.insert_views(conn, view_lines)
    utils.insert_ratings(conn, rate_lines)
    utils.insert_recommendations(conn, recommendation_lines)
    utils.insert_errors(conn, error_lines)

    conn.commit()
    conn.close()
    return len(view_lines), len(rate_lines), len(recommendation_lines), len(error_lines)

def stream():
    try:
        consumer = KafkaConsumer(
        config.KAFKA_TOPIC,
        bootstrap_servers=[config.KAFKA_SERVER],
        auto_offset_reset='earliest', #Experiment with different values
        # Commit that an offset has been read
        enable_auto_commit=True,
        # How often to tell Kafka, an offset has been read
        auto_commit_interval_ms=1000,
        group_id=config.KAFKA_GROUP
        )

        # Utilize a buffer so we can process more with a single connection
        buffer = []
        size = 1000
        for message in consumer:
            message = message.value.decode()
            # Default message.value type is bytes!
            buffer.append(message)
            if len(buffer) == size:
                s = time.time()
                views, rates, reccs, errors = database_insert(buffer)
                duration = time.time()-s
                print(f"""Inserted {size} lines in: {duration} seconds. Avg of {size / duration} per second
                        Views: {views}, Rates: {rates}, Reccs: {reccs}, Errors: {errors}""")
                buffer = []
              
    finally:
        consumer.close()

def main():
    parser = argparse.ArgumentParser(
                    prog='Stream',
                    description='Stream Kafka Data to Postgres')
    
    # Add argument for database choice
    parser.add_argument(
        "--database",
        choices=["prod", "dev"],
        default="dev",
        help="Specify whether to write to prod or dev database (default: dev)"
    )

    stream()

if __name__ == "__main__":
    main()