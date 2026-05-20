#!/usr/bin/env python3
"""
ai_firewall.py — Módulo 4
Motor de detección y bloqueo en tiempo real.
Captura tráfico → extrae 13 features → clasifica → bloquea con nftables.
Entregable E-06
"""

import subprocess
import logging
import sqlite3
import joblib
import numpy as np
import os
import sys
import time
import signal
from datetime import datetime
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

# ── Configuración ────────────────────────────────────────────────
IFACE         = 'enp0s3'          # interfaz de red a monitorear
MODEL_PATH    = 'models/firewall_ai_model.joblib'
SCALER_PATH   = 'models/scaler.joblib'
DB_PATH       = 'data/firewall_metrics.db'
LOG_PATH      = 'data/ai_firewall.log'
WINDOW_SEC    = 5        # ventana de análisis por IP (segundos)
THRESHOLD     = 0.17      # umbral de probabilidad para bloqueo
MAX_RATE      = 100       # pkts/seg para activar potential_flood

# ── Whitelist — estas IPs NUNCA se bloquean (RNF-07) ────────────
WHITELIST = {
    '127.0.0.1',
    '192.168.100.10',   # propio servidor
    '192.168.100.1',    # gateway
    '192.168.56.1',     # tu PC (host-only)
}

# ── Features (mismo orden que el entrenamiento) ──────────────────
FEATURE_COLS = [
    'total_pkts', 'tcp_pkts', 'udp_pkts', 'other_pkts',
    'unique_dports_count', 'syn_ratio', 'avg_pkt_size',
    'duration_sec', 'bytes_per_sec', 'port_scan_score',
    'small_syn_score', 'potential_flood', 'potential_scan'
]

# ── Logging estructurado — RNF-08 ───────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger('ai-firewall')


class PacketBuffer:
    """Acumula paquetes por IP en ventanas de tiempo"""
    def __init__(self):
        self.data = defaultdict(lambda: {
            'pkts': [], 'sizes': [], 'timestamps': [],
            'tcp': 0, 'udp': 0, 'other': 0,
            'dports': set(), 'syn': 0, 'small_syn': 0
        })

    def add(self, pkt):
        if not pkt.haslayer(IP):
            return
        src = pkt[IP].src
        now = time.time()
        size = len(pkt)
        d = self.data[src]
        d['pkts'].append(pkt)
        d['sizes'].append(size)
        d['timestamps'].append(now)

        if pkt.haslayer(TCP):
            d['tcp'] += 1
            d['dports'].add(pkt[TCP].dport)
            if pkt[TCP].flags == 0x02:
                d['syn'] += 1
                if size < 60:
                    d['small_syn'] += 1
        elif pkt.haslayer(UDP):
            d['udp'] += 1
            d['dports'].add(pkt[UDP].dport)
        else:
            d['other'] += 1

    def extract_features(self, src_ip):
        """Extrae las 13 features para una IP"""
        d = self.data[src_ip]
        total = len(d['pkts'])
        if total == 0:
            return None

        ts = d['timestamps']
        dur = max(ts) - min(ts) if len(ts) > 1 else 0.001
        total_bytes = sum(d['sizes'])
        tcp = d['tcp']

        return np.array([[
            total,
            tcp,
            d['udp'],
            d['other'],
            len(d['dports']),
            round(d['syn'] / max(tcp, 1), 4),
            round(np.mean(d['sizes']), 2),
            round(dur, 4),
            round(total_bytes / dur, 2),
            min(len(d['dports']) / 100, 1.0),
            round(d['small_syn'] / max(tcp, 1), 4),
            1 if (total / dur) > MAX_RATE else 0,
            1 if len(d['dports']) > 50 else 0,
        ]])

    def flush_old(self, window=WINDOW_SEC):
        """Limpia IPs con datos más antiguos que la ventana"""
        now = time.time()
        to_del = []
        for ip, d in self.data.items():
            if d['timestamps'] and (now - d['timestamps'][-1]) > window:
                to_del.append(ip)
        for ip in to_del:
            del self.data[ip]

    def get_ips(self):
        return list(self.data.keys())


class FirewallController:
    """Gestiona el bloqueo/desbloqueo de IPs en nftables"""

    def __init__(self):
        self.blocked = set()

    def block(self, ip):
        if ip in self.blocked or ip in WHITELIST:
            return False
        cmd = ['sudo', 'nft', 'add', 'element', 'inet', 'filter',
               'ia_blocklist', '{', ip, '}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self.blocked.add(ip)
            log.warning(f"BLOQUEADO | ip={ip}")
            return True
        else:
            log.error(f"Error bloqueando {ip}: {result.stderr}")
            return False

    def unblock(self, ip):
        cmd = ['sudo', 'nft', 'delete', 'element', 'inet', 'filter',
               'ia_blocklist', '{', ip, '}']
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self.blocked.discard(ip)
            log.info(f"DESBLOQUEADO | ip={ip}")
            return True
        return False


class MetricsDB:
    """Guarda cada decisión en SQLite para Grafana — RNF-08"""

    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                src_ip    TEXT,
                action    TEXT,
                prob_attack REAL,
                total_pkts  INTEGER,
                syn_ratio   REAL,
                port_scan_score REAL
            )
        ''')
        self.conn.commit()

    def insert(self, src_ip, action, prob_attack, features):
        self.conn.execute('''
            INSERT INTO decisions
            (timestamp, src_ip, action, prob_attack, total_pkts, syn_ratio, port_scan_score)
            VALUES (?,?,?,?,?,?,?)
        ''', (
            datetime.now().isoformat(),
            src_ip, action,
            round(prob_attack, 4),
            int(features[0][0]),
            round(float(features[0][5]), 4),
            round(float(features[0][9]), 4),
        ))
        self.conn.commit()


class AIFirewall:
    """Sistema principal — detect-and-respond"""

    def __init__(self):
        log.info("Iniciando AIFirewall...")
        self.model     = joblib.load(MODEL_PATH)
        self.scaler    = joblib.load(SCALER_PATH)
        self.buffer    = PacketBuffer()
        self.firewall  = FirewallController()
        self.db        = MetricsDB(DB_PATH)
        self.running   = True
        log.info(f"Modelo cargado: {MODEL_PATH}")
        log.info(f"Interfaz:       {IFACE}")
        log.info(f"Whitelist:      {WHITELIST}")
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

    def _shutdown(self, *args):
        log.info("Apagando AIFirewall (fail-safe: nftables sigue activo)")
        self.running = False
        sys.exit(0)

    def analyze(self):
        """Analiza IPs acumuladas en el buffer y decide bloqueo"""
        for src_ip in self.buffer.get_ips():
            if src_ip in WHITELIST:
                continue
            if src_ip in self.firewall.blocked:
                continue

            features = self.buffer.extract_features(src_ip)
            if features is None:
                continue

            features_sc   = self.scaler.transform(features)
            prob_attack   = self.model.predict_proba(features_sc)[0][1]
            prediction    = 1 if prob_attack >= THRESHOLD else 0

            action = 'BLOCK' if prediction == 1 else 'ALLOW'

            # Log estructurado ISO-8601 — RNF-08
            log.info(
                f"DECISION | ip={src_ip} | action={action} | "
                f"prob={prob_attack:.4f} | "
                f"pkts={int(features[0][0])} | "
                f"syn_ratio={features[0][5]:.3f} | "
                f"ports={int(features[0][4])}"
            )

            # Guardar en SQLite para Grafana
            self.db.insert(src_ip, action, prob_attack, features)

            if prediction == 1:
                blocked = self.firewall.block(src_ip)
                if blocked:
                    log.warning(
                        f"BLOQUEADO | ip={src_ip} | "
                        f"prob_attack={prob_attack:.4f}"
                    )

        self.buffer.flush_old()

    def packet_callback(self, pkt):
        """Callback invocado por scapy por cada paquete capturado"""
        self.buffer.add(pkt)

    def run(self):
        log.info("=" * 50)
        log.info("AIFirewall activo — monitoreando tráfico")
        log.info("=" * 50)

        last_analysis = time.time()

        while self.running:
            # Capturar paquetes en ráfagas de 5 segundos
            sniff(
                iface=IFACE,
                prn=self.packet_callback,
                store=False,
                timeout=5
            )
            # Analizar cada WINDOW_SEC segundos
            if time.time() - last_analysis >= WINDOW_SEC:
                self.analyze()
                last_analysis = time.time()


if __name__ == '__main__':
    # Verificar que los modelos existen
    for path in [MODEL_PATH, SCALER_PATH]:
        if not os.path.exists(path):
            log.error(f"No se encuentra: {path} — ejecuta train_model.py primero")
            sys.exit(1)

    ai = AIFirewall()
    ai.run()
