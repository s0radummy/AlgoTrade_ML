"""
Standalone Kafka round-trip test.
Does NOT require Redis, InfluxDB, or TFT model — pure Kafka only.

Usage:
    python scripts/test_kafka.py
    python scripts/test_kafka.py --bootstrap localhost:29092  # if using host port
"""

import argparse
import json
import time
import uuid
from datetime import datetime

from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError

BOOTSTRAP_DEFAULT = "localhost:29092"
TEST_TOPIC = f"kafka-test-{uuid.uuid4().hex[:8]}"
NUM_MESSAGES = 10
CONSUME_TIMEOUT_MS = 15000


def print_result(label: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def create_topic(bootstrap: str):
    admin = KafkaAdminClient(bootstrap_servers=[bootstrap], request_timeout_ms=10000)
    topic = NewTopic(name=TEST_TOPIC, num_partitions=1, replication_factor=1)
    try:
        admin.create_topics([topic])
        print(f"       Created topic: {TEST_TOPIC}")
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def delete_topic(bootstrap: str):
    try:
        admin = KafkaAdminClient(bootstrap_servers=[bootstrap], request_timeout_ms=10000)
        admin.delete_topics([TEST_TOPIC])
        admin.close()
    except Exception:
        pass


def run_tests(bootstrap: str):
    print(f"\nKafka round-trip test  |  broker: {bootstrap}\n{'='*50}")
    results = []

    # --- 1. Broker connectivity ---
    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            request_timeout_ms=10000,
        )
        results.append(print_result("Broker connectivity", True))
    except Exception as e:
        results.append(print_result("Broker connectivity", False, str(e)))
        print("\nCannot reach broker — is 'docker-compose up -d kafka zookeeper' running?")
        return results

    # --- 2. Topic creation ---
    try:
        create_topic(bootstrap)
        results.append(print_result("Topic creation", True, TEST_TOPIC))
    except Exception as e:
        results.append(print_result("Topic creation", False, str(e)))

    # --- 3. Produce messages ---
    sent = []
    try:
        for i in range(NUM_MESSAGES):
            payload = {
                "seq": i,
                "symbol": "TEST",
                "last_price": 100.0 + i,
                "volume": 1000 * (i + 1),
                "timestamp": datetime.utcnow().isoformat(),
            }
            future = producer.send(TEST_TOPIC, value=payload, key=b"TEST")
            meta = future.get(timeout=10)
            sent.append((meta.partition, meta.offset))

        producer.flush()
        producer.close()
        results.append(print_result("Produce messages", True, f"{NUM_MESSAGES} messages sent"))
    except Exception as e:
        results.append(print_result("Produce messages", False, str(e)))
        return results

    # --- 4. Consume & verify ---
    try:
        consumer = KafkaConsumer(
            TEST_TOPIC,
            bootstrap_servers=[bootstrap],
            group_id=f"test-grp-{uuid.uuid4().hex[:6]}",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=CONSUME_TIMEOUT_MS,
            max_poll_records=50,
        )

        received = []
        for msg in consumer:
            received.append(msg.value)
            if len(received) >= NUM_MESSAGES:
                break

        consumer.close()

        ok = len(received) == NUM_MESSAGES
        results.append(print_result("Consume messages", ok, f"got {len(received)}/{NUM_MESSAGES}"))

        # Verify ordering / content
        seqs = [m["seq"] for m in received]
        ordered = seqs == sorted(seqs)
        results.append(print_result("Message ordering", ordered, f"seqs={seqs}"))

        # Verify last_price values
        prices_ok = all(m["last_price"] == 100.0 + m["seq"] for m in received)
        results.append(print_result("Payload integrity", prices_ok))

    except Exception as e:
        results.append(print_result("Consume messages", False, str(e)))

    # --- 5. Latency smoke-test (single message RTT) ---
    try:
        lat_topic = f"lat-{uuid.uuid4().hex[:6]}"
        admin = KafkaAdminClient(bootstrap_servers=[bootstrap])
        admin.create_topics([NewTopic(name=lat_topic, num_partitions=1, replication_factor=1)])
        admin.close()

        prod = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        cons = KafkaConsumer(
            lat_topic,
            bootstrap_servers=[bootstrap],
            group_id=f"lat-grp-{uuid.uuid4().hex[:6]}",
            auto_offset_reset="earliest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=10000,
        )

        t0 = time.perf_counter()
        prod.send(lat_topic, value={"ts": t0}).get(timeout=5)
        prod.flush()
        prod.close()

        for msg in cons:
            rtt_ms = (time.perf_counter() - t0) * 1000
            break

        cons.close()
        KafkaAdminClient(bootstrap_servers=[bootstrap]).delete_topics([lat_topic])

        ok = rtt_ms < 1000
        results.append(print_result("Single-message RTT", ok, f"{rtt_ms:.1f} ms"))
    except Exception as e:
        results.append(print_result("Single-message RTT", False, str(e)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default=BOOTSTRAP_DEFAULT,
                        help=f"Kafka bootstrap server (default: {BOOTSTRAP_DEFAULT})")
    args = parser.parse_args()

    results = run_tests(args.bootstrap)

    # Cleanup
    try:
        delete_topic(args.bootstrap)
    except Exception:
        pass

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Kafka architecture is working correctly.")
    else:
        print("Some checks failed — see details above.")


if __name__ == "__main__":
    main()
