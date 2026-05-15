# Kafka

## Links

- Architecture - https://www.geeksforgeeks.org/apache-kafka/kafka-architecture/
- GitHub - https://gist.github.com/piyushgarg-dev/32cadf6420c452b66a9a6d977ade0b01
- Video - https://www.youtube.com/watch?v=ZJJHm_bd9Zo

---

## Use-Cases

## `Uber/Zomato Feature`

- Driver location gets updated in real-time
- Assume 1k drivers at a single time
- Updating Location, ETA, etc. would require >2k ops
- `ops = operations per second`
- Databases fail in these situations

---

## Problem & Solution

- Databases have **Low Throughput** & **High Storage**
- Kafka has **High Throughput** & **Low Storage**

---

## Structure (Taking Uber as Example)

- 100k cars
- Speed, location, ETA updated every second (**PRODUCER**)
- Information goes to Kafka (temporary storage type)
- Fare, Analytics, Customer services, etc. are **SERVERS**
- Data gets **BULK-INSERTED** into the database (better for low-throughput databases)

---

## Concepts

#### TOPIC

- **Logical sectioning** of **messages** in the Kafka **server**

#### PARTITION

- Topics can also contain very large amounts of data
- So, they are divided further into **partitions**

#### AUTO-BALANCING

- info-inflow = (info-outflow) / group

#### Notes

- 1 consumer can consume multiple partitions
- But 1 partition can only supply to 1 consumer **per group**

#### CONSUMER GROUPS

- Check:
  - Queue Architecture (**RabbitMQ, SQS**)
  - Pub/Sub Architecture
- Kafka can act as both:
  - A Queue
  - A Pub/Sub system

#### ZOOKEEPER

- Kafka uses this service internally
