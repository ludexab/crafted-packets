from scapy.all import *
from scapy.layers.inet import IP, TCP
import time

# Define the IP and port of the RabbitMQ server
target_ip = '127.0.0.1'  # RabbitMQ server IP
target_port = 5672       # AMQP port (usually 5672 for non-SSL, 5671 for SSL)

# Create a custom TCP packet
ip_packet = IP(dst=target_ip)
tcp_packet = TCP(dport=target_port, sport=RandShort(), flags="S")

# Crafting a malicious and a legitimate AMQP-like message (fake frame)
malicious_payload = b'\x01\x02\x03\x04MaliciousPayload'
legit_payload = b'\x00\x00\x00\xCE\x00\x0A\x08\x01\x01\x02UnauthorizedMsg'

# Combine the layers into one packet
packet = ip_packet / tcp_packet / malicious_payload
packet2 = ip_packet / tcp_packet / legit_payload

# Send the packets in a loop
while True:
    send(packet)
    send(packet2)
    time.sleep(0.1)
