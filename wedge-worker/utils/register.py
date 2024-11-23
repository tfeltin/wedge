import socket
import time


def register(config):
    # Announce oneself to the orchestrator
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    while True:
        try:
            sock.connect((config['orchestrator'], int(config['registration_port'])))
            sock.settimeout(None)
            sock.sendall(b"SYN")
            break
        except Exception as e:
            print("Wedge Worker -- Failed connecting to %s: %s" % (config['orchestrator'], str(e)))
            time.sleep(1)
    sock.close()
