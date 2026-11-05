import argparse
import base64
import hashlib
import ipaddress
import os
import re
import secrets
import socket
import subprocess
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2

PORTS = {80: "http", 443: "https", 554: "rtsp", 2020: "onvif", 5543: "rtsp", 6668: "tuya", 8000: "onvif", 8080: "http", 8554: "rtsp", 8899: "onvif"}
RTSP_PORTS = (554, 5543, 8554)
ONVIF_PORTS = (2020, 8000, 8899)
CAMERA_FILE = Path(".camera_url")
RTSP_PATHS = ["", "stream1", "stream2", "live", "live/ch0", "live/ch00_0", "ch0_0.h264", "h264_stream", "onvif1", "11", "12", "Streaming/Channels/101", "cam/realmonitor?channel=1&subtype=0", "user=admin_password=_channel=1_stream=0.sdp"]
PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{id}</w:MessageID><w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To><w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""


def wsse_header(user, password):
    if not user:
        return ""
    nonce = secrets.token_bytes(16)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
    return (
        '<s:Header><Security xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"><UsernameToken>'
        f"<Username>{user}</Username>"
        f'<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password>'
        f'<Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>'
        f'<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created>'
        "</UsernameToken></Security></s:Header>"
    )


def onvif_call(url, body, user, password):
    envelope = f'<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope">{wsse_header(user, password)}<s:Body>{body}</s:Body></s:Envelope>'
    status, text = onvif_post(urllib.request.build_opener(), url, envelope)
    if user and status in (400, 401):
        handler = urllib.request.HTTPDigestAuthHandler()
        handler.add_password(None, url, user, password)
        plain = f'<?xml version="1.0"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"><s:Body>{body}</s:Body></s:Envelope>'
        status, text = onvif_post(urllib.request.build_opener(handler), url, plain)
    return status, text


def onvif_post(opener, url, envelope):
    request = urllib.request.Request(url, envelope.encode(), {"Content-Type": "application/soap+xml"})
    try:
        with opener.open(request, timeout=5) as response:
            return response.status, response.read().decode(errors="ignore")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="ignore")


def onvif_stream_uris(ip, port, user, password):
    url = f"http://{ip}:{port}/onvif/device_service"
    status, text = onvif_call(url, '<GetProfiles xmlns="http://www.onvif.org/ver10/media/wsdl"/>', user, password)
    if status != 200:
        return status, []
    uris = []
    for token in re.findall(r'<[^>]*Profiles[^>]*token="([^"]+)"', text):
        body = (
            '<GetStreamUri xmlns="http://www.onvif.org/ver10/media/wsdl"><StreamSetup>'
            '<Stream xmlns="http://www.onvif.org/ver10/schema">RTP-Unicast</Stream>'
            '<Transport xmlns="http://www.onvif.org/ver10/schema"><Protocol>RTSP</Protocol></Transport>'
            f"</StreamSetup><ProfileToken>{token}</ProfileToken></GetStreamUri>"
        )
        status, text = onvif_call(url, body, user, password)
        uris += re.findall(r"<[^>]*Uri>([^<]+)</[^>]*Uri>", text)
    return 200, uris


def with_credentials(url, user, password):
    return url.replace("rtsp://", f"rtsp://{user}:{password}@", 1) if user else url


def local_network():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("192.0.2.1", 9))
        ip = s.getsockname()[0]
    return ipaddress.ip_network(f"{ip}/24", strict=False), ip


def discover_onvif(local_ip, timeout=3.0):
    found = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local_ip))
        s.settimeout(timeout)
        s.sendto(PROBE.format(id=uuid.uuid4()).encode(), ("239.255.255.250", 3702))
        try:
            while True:
                data, (ip, _) = s.recvfrom(65535)
                found[ip] = re.findall(r"XAddrs>([^<]+)<", data.decode(errors="ignore"))
        except TimeoutError:
            pass
    return found


def is_open(ip, port, timeout=2.0):
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((str(ip), port)) == 0


def scan(hosts):
    pairs = [(ip, port) for ip in hosts for port in PORTS]
    with ThreadPoolExecutor(64) as pool:
        states = pool.map(lambda pair: is_open(*pair), pairs)
    found = {}
    for (ip, port), state in zip(pairs, states):
        if state:
            found.setdefault(str(ip), []).append(port)
    return found


def rtsp_answer(ip, port, path, user, password):
    credentials = f"{user}:{password}@" if user else ""
    url = f"rtsp://{credentials}{ip}:{port}/{path}"
    result = subprocess.run(["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-timeout", "4000000", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", url], capture_output=True, text=True)
    return url, result.stdout.strip(), result.stderr.strip()


def show_url(url, password):
    url = re.sub(r"/[0-9a-f]{32}/", "/***/", url)
    return url.replace(f":{password}@", ":***@") if password else url


def tuya_url(ip, port, user, password):
    token = hashlib.md5(f"{user}{password}".encode()).hexdigest()
    return f"rtsp://{ip}:{port}/{token}/live/channel0"


def try_tuya(ip, port, user, password):
    url = tuya_url(ip, port, user, password)
    result = subprocess.run(["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-timeout", "6000000", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", url], capture_output=True, text=True)
    print(f"Caminho Tuya em {ip}:{port}: {result.stdout.strip() or show_url(result.stderr.strip(), password)}")
    return url if result.stdout.strip() else None


def save_url(url):
    CAMERA_FILE.touch(mode=0o600)
    CAMERA_FILE.write_text(url)


def try_rtsp(ip, port, user, password):
    print(f"Testando caminhos RTSP em {ip}:{port}...")
    for path in RTSP_PATHS:
        url, stream, error = rtsp_answer(ip, port, path, user, password)
        if stream:
            print(f"  FUNCIONA: {show_url(url, password)} ({stream})")
            return url
        if "401" in error or "403" in error:
            print("  A câmera recusou: usuário ou senha incorretos." if user else "  A câmera pediu usuário e senha. Rode de novo com --user e --password.")
            return None
    print("  A porta está aberta, mas nenhum caminho conhecido respondeu com vídeo.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Procura câmeras na rede local e testa se aceitam ONVIF ou RTSP")
    parser.add_argument("--ip", help="IP da câmera; sem ele procura em toda a rede local")
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default=os.environ.get("CAMERA_PASSWORD", ""), help="também pode vir da variável CAMERA_PASSWORD")
    args = parser.parse_args()

    network, local_ip = local_network()
    print(f"Rede local: {network}")

    onvif = discover_onvif(local_ip)
    for ip, addresses in onvif.items():
        print(f"ONVIF respondeu em {ip}: {' '.join(addresses)}")
    if not onvif:
        print("Nenhum dispositivo respondeu à busca ONVIF.")

    hosts = [ipaddress.ip_address(args.ip)] if args.ip else [h for h in network.hosts() if str(h) != local_ip]
    found = scan(hosts)
    if not found:
        print("Nenhuma porta conhecida aberta.")
    for ip, ports in found.items():
        print(f"{ip}: " + ", ".join(f"{p} ({PORTS[p]})" for p in ports))

    urls = []
    for ip, ports in found.items():
        for port in ports:
            if port in ONVIF_PORTS:
                status, uris = onvif_stream_uris(ip, port, args.user, args.password)
                if status != 200:
                    print(f"ONVIF em {ip}:{port} recusou o acesso ({status}): " + ("usuário ou senha incorretos." if args.user else "ela pede usuário e senha."))
                for uri in uris:
                    print(f"ONVIF informou o stream {uri}")
                    urls.append(with_credentials(uri, args.user, args.password))
    if not urls and args.user:
        urls = [try_tuya(ip, port, args.user, args.password) for ip, ports in found.items() for port in ports if port == 5543]
        urls = [u for u in urls if u]
    if not urls:
        urls = [try_rtsp(ip, port, args.user, args.password) for ip, ports in found.items() for port in ports if port in RTSP_PORTS]
        urls = [u for u in urls if u]
    for url in urls:
        ok, frame = cv2.VideoCapture(url, cv2.CAP_FFMPEG).read()
        print(f"Quadro lido de {show_url(url, args.password)}: {frame.shape[1]}x{frame.shape[0]}" if ok else f"Sem quadro de {show_url(url, args.password)}")
        if ok:
            save_url(url)
            print(f"URL salva em {CAMERA_FILE}. Para contar: uv run 04_count_webcam.py --source camera")
    if not urls:
        print("Nenhum stream RTSP encontrado.")


if __name__ == "__main__":
    main()
