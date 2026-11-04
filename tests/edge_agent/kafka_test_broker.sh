#!/bin/bash
# Throw-away single-node Kafka (KRaft) with TLS + SASL (PLAIN, SCRAM-SHA-512) and an ACL authorizer,
# for tests/edge_agent/test_source_kafka_live.py. Rootless podman; binds 127.0.0.1 only.
#
#   kafka_test_broker.sh up   DIR PORT     start; writes DIR/{ca.pem,client.pem,client.key,env} (secrets are random, 0600)
#   kafka_test_broker.sh down DIR          stop and remove the container, delete DIR
#
# Listeners (all advertised on 127.0.0.1):  PORT   SASL_SSL (PLAIN / SCRAM-SHA-512; users admin, edge)
#                                           PORT+1 SSL with a required client certificate (CN=edge-client)
# `admin` and CN=edge-client are super users; `edge` has no ACL until a test grants it one.
# Then: export EDGE_KAFKA_TEST_DIR=DIR
set -euo pipefail
IMAGE="${KAFKA_IMAGE:-docker.io/apache/kafka:3.9.1}"     # sha256:44a48f9d78af26959fa4d61accb9be35f939a79bfdae05cad6df4baca8e22eeb
cmd="${1:?up|down}"; DIR="${2:?dir}"; NAME="agdkafka1029-$(basename "$DIR" | tr -c 'A-Za-z0-9\n' _)"
case "$cmd" in
down) podman rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$DIR"; echo "down: $NAME"; exit 0 ;;
up) ;;
*) echo "usage: $0 up DIR PORT | down DIR" >&2; exit 2 ;;
esac
PORT="${3:?port}"; TPORT=$((PORT + 1))
mkdir -p "$DIR"; chmod 700 "$DIR"; cd "$DIR"
rand() { head -c 18 /dev/urandom | base64 | tr -d '/+=' ; }
CLUSTER_ID="$(head -c 16 /dev/urandom | base64 | tr "+/" "-_" | cut -c1-22)"   # fixed, so a container restart re-formats nothing
ADMIN_PASS="$(rand)"; EDGE_PASS="$(rand)"; CLIENT_KEY_PASS="$(rand)"
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 3 -subj "/CN=edge-kafka-test-ca" 2>/dev/null
mk() { # name subj san
  openssl req -newkey rsa:2048 -nodes -keyout "$1.key.rsa" -out "$1.csr" -subj "$2" 2>/dev/null
  printf "subjectAltName=%s\n" "$3" > "$1.ext"
  openssl x509 -req -in "$1.csr" -CA ca.pem -CAkey ca.key -CAcreateserial -out "$1.crt" -days 3 -extfile "$1.ext" 2>/dev/null
  openssl pkcs8 -topk8 -nocrypt -in "$1.key.rsa" -out "$1.key" 2>/dev/null
}
mk broker "/CN=localhost" "IP:127.0.0.1,DNS:localhost"
mk client "/CN=edge-client" "DNS:edge-client"
cat broker.key broker.crt > broker-keystore.pem
cp client.crt client.pem
# an encrypted copy of the client key, for the key_password_env test
openssl pkcs8 -topk8 -in client.key.rsa -out client-enc.key -passout "pass:$CLIENT_KEY_PASS" 2>/dev/null
cat > server.properties <<P
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9094
controller.listener.names=CONTROLLER
listeners=SASL_SSL://:9093,SSL://:9095,CONTROLLER://localhost:9094,BROKER://localhost:9092
advertised.listeners=SASL_SSL://127.0.0.1:$PORT,SSL://127.0.0.1:$TPORT,BROKER://localhost:9092
inter.broker.listener.name=BROKER
listener.security.protocol.map=SASL_SSL:SASL_SSL,SSL:SSL,CONTROLLER:PLAINTEXT,BROKER:PLAINTEXT
log.dirs=/tmp/kraft
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
group.initial.rebalance.delay.ms=0
group.min.session.timeout.ms=6000
auto.create.topics.enable=false
ssl.keystore.type=PEM
ssl.keystore.location=/etc/kafka-test/broker-keystore.pem
ssl.truststore.type=PEM
ssl.truststore.location=/etc/kafka-test/ca.pem
listener.name.ssl.ssl.client.auth=required
sasl.enabled.mechanisms=PLAIN,SCRAM-SHA-512
listener.name.sasl_ssl.plain.sasl.jaas.config=org.apache.kafka.common.security.plain.PlainLoginModule required user_admin="$ADMIN_PASS" user_edge="$EDGE_PASS";
listener.name.sasl_ssl.scram-sha-512.sasl.jaas.config=org.apache.kafka.common.security.scram.ScramLoginModule required;
authorizer.class.name=org.apache.kafka.metadata.authorizer.StandardAuthorizer
allow.everyone.if.no.acl.found=false
super.users=User:admin;User:CN=edge-client;User:ANONYMOUS
P
cat > env <<E
EDGE_KAFKA_BOOTSTRAP=127.0.0.1:$PORT
EDGE_KAFKA_TLS_BOOTSTRAP=127.0.0.1:$TPORT
EDGE_KAFKA_ADMIN_USER=admin
EDGE_KAFKA_ADMIN_PASSWORD=$ADMIN_PASS
EDGE_KAFKA_USER=edge
EDGE_KAFKA_PASSWORD=$EDGE_PASS
EDGE_KAFKA_CLIENT_KEY_PASSWORD=$CLIENT_KEY_PASS
E
chmod 600 env *.key *.key.rsa ca.key broker-keystore.pem server.properties
chmod 644 ca.pem client.pem client.key client-enc.key broker-keystore.pem server.properties   # the container user reads these
chmod 600 client.key client-enc.key
chmod 644 broker-keystore.pem server.properties; chmod 755 "$DIR"
podman rm -f "$NAME" >/dev/null 2>&1 || true
podman run -d --name "$NAME" --cpus 1.5 --memory 1200m -e KAFKA_HEAP_OPTS="-Xms256m -Xmx600m" \
  -p "127.0.0.1:$PORT:9093" -p "127.0.0.1:$TPORT:9095" -v "$DIR:/etc/kafka-test:ro,Z" \
  --entrypoint /bin/bash "$IMAGE" -c "
    set -e
    /opt/kafka/bin/kafka-storage.sh format -t $CLUSTER_ID -c /etc/kafka-test/server.properties --ignore-formatted \
      --add-scram 'SCRAM-SHA-512=[name=admin,password=$ADMIN_PASS]' --add-scram 'SCRAM-SHA-512=[name=edge,password=$EDGE_PASS]' >/dev/null
    exec /opt/kafka/bin/kafka-server-start.sh /etc/kafka-test/server.properties" >/dev/null
for _ in $(seq 60); do
  if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then sleep 3; echo "up: $NAME on 127.0.0.1:$PORT (tls+client-cert on $TPORT); dir $DIR"; exit 0; fi
  sleep 1
done
podman logs "$NAME" | tail -20; exit 1
