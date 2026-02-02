#!/usr/bin/env python3
"""Simple test that connects to server and verifies render works."""
import socket
import struct
import json
import sys

def send_msg(sock, obj):
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)

def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf

def recv_msg(sock):
    header = recv_exact(sock, 4)
    (n,) = struct.unpack("!I", header)
    payload = recv_exact(sock, n)
    return json.loads(payload.decode("utf-8"))

# Use port 5556 for this test
HOST = "127.0.0.1"
PORT = 5556

print(f"Connecting to {HOST}:{PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((HOST, PORT))
    print("✓ Connected")
    
    print("\nSending render command...")
    send_msg(sock, {"cmd": "render"})
    print("✓ Command sent")
    
    print("\nWaiting for response...")
    resp = recv_msg(sock)
    
    if resp.get("ok"):
        print(f"✓ Render successful!")
        print(f"  Width: {resp['width']}")
        print(f"  Height: {resp['height']}")
        print(f"  RGBA PNG size: {len(resp['rgba_png_b64'])} bytes (base64)")
        print(f"  Depth data size: {len(resp['depth_f32_zlib_b64'])} bytes (base64)")
        print(f"  Index data size: {len(resp['index_i32_zlib_b64'])} bytes (base64)")
        
        # Close connection
        send_msg(sock, {"cmd": "close"})
        print("\n✓ All tests passed!")
    else:
        print(f"✗ Render failed: {resp.get('error')}")
        sys.exit(1)
        
except ConnectionRefusedError:
    print(f"✗ Connection refused. Is the server running on port {PORT}?")
    print("\nStart the server with:")
    print(f"  ./start_server.sh")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    sock.close()
