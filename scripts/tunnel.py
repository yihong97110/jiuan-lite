# -*- coding: utf-8 -*-
"""本地端口转发隧道: 127.0.0.1:<LOCAL> -> (SSH) -> 服务器 127.0.0.1:<REMOTE>。
用 paramiko，无需在命令行传密码给系统 ssh。后台常驻。
"""
import select
import socket
import sys
import threading

import paramiko

HOST, USER, PW = "150.158.58.93", "ubuntu", "tiancaiping38@gmail.com"
LOCAL_PORT = 8888
REMOTE_HOST, REMOTE_PORT = "127.0.0.1", 8000


class Handler(threading.Thread):
    def __init__(self, sock, transport):
        super().__init__(daemon=True)
        self.sock = sock
        self.transport = transport

    def run(self):
        try:
            chan = self.transport.open_channel(
                "direct-tcpip", (REMOTE_HOST, REMOTE_PORT), self.sock.getpeername()
            )
        except Exception as e:
            print("channel open failed:", e, flush=True)
            self.sock.close()
            return
        if chan is None:
            self.sock.close()
            return
        while True:
            r, _, _ = select.select([self.sock, chan], [], [])
            if self.sock in r:
                data = self.sock.recv(4096)
                if len(data) == 0:
                    break
                chan.sendall(data)
            if chan in r:
                data = chan.recv(4096)
                if len(data) == 0:
                    break
                self.sock.sendall(data)
        chan.close()
        self.sock.close()


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PW, timeout=20,
                   look_for_keys=False, allow_agent=False)
    transport = client.get_transport()
    transport.set_keepalive(30)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LOCAL_PORT))
    srv.listen(50)
    print(f"TUNNEL_UP 127.0.0.1:{LOCAL_PORT} -> {HOST}:{REMOTE_PORT}", flush=True)
    while True:
        try:
            sock, _ = srv.accept()
        except OSError:
            break
        Handler(sock, transport).start()


if __name__ == "__main__":
    main()
