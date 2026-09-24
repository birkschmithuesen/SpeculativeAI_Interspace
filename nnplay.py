import sys, time
from pythonosc import udp_client
c = udp_client.SimpleUDPClient("127.0.0.1", 8001)
c.send_message("/NN/play", 1)
c.send_message("/colors/NN/red", 1.0)
c.send_message("/colors/NN/green", 1.0)
c.send_message("/colors/NN/blue", 1.0)
print("NN/play=1 und Farbe weiss gesendet")
