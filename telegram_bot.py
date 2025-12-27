#!/usr/bin/env python3
"""
Bot Telegram per monitorare l'output di un processo Python di lunga durata
e inviare aggiornamenti al canale Telegram.
"""

import subprocess
import sys
import time
import requests
import argparse
import re
import os
from datetime import datetime


class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, text, max_retries=3):
        """Invia un messaggio al canale Telegram con retry logic."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    print(f"[BOT] Messaggio inviato: {text[:50]}...")
                    return True
                else:
                    print(f"[BOT] Errore invio (tentativo {attempt + 1}): {response.status_code} - {response.text}")
            except Exception as e:
                print(f"[BOT] Errore connessione (tentativo {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def send_startup_message(self, script_name):
        """Invia un messaggio di avvio."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"🚀 <b>Bot avviato</b>\n"
        message += f"⏰ {timestamp}\n"
        message += f"📄 Script: {script_name}"
        self.send_message(message)
    
    def send_completion_message(self, script_name, exit_code):
        """Invia un messaggio di completamento."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "✅ Completato" if exit_code == 0 else f"❌ Errore (exit code: {exit_code})"
        message = f"{status}\n"
        message += f"⏰ {timestamp}\n"
        message += f"📄 Script: {script_name}"
        self.send_message(message)


def monitor_process(bot, script_path, args=None, is_module=False):
    """
    Esegue il processo Python e monitora l'output in tempo reale.
    Invia messaggi Telegram quando trova pattern specifici.
    """
    # Costruisci il comando
    if is_module:
        cmd = [sys.executable, "-m", script_path]
    else:
        cmd = [sys.executable, script_path]
    
    if args:
        cmd.extend(args)
    
    print(f"[BOT] Avvio processo: {' '.join(cmd)}")
    bot.send_startup_message(script_path)
    
    try:
        # Configura l'environment per supportare UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUNBUFFERED'] = '1'
        
        # Avvia il processo
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        # Monitora l'output riga per riga in tempo reale
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            # Stampa l'output anche localmente (con flush per output immediato)
            print(line, end='', flush=True)
            
            line_stripped = line.strip()
            
            # Controlla se la riga contiene "Move 00"
            if "Move 00" in line_stripped:
                bot.send_message(f"🎮 {line_stripped}")
            
            # Controlla se la riga contiene "NUOVA EPOCA"
            # Supporta formati come "NUOVA EPOCA: 5" o "NUOVA EPOCA 10"
            match = re.search(r'NUOVA EPOCA[:\s]+(\d+)', line_stripped, re.IGNORECASE)
            if match:
                epoca = match.group(1)
                bot.send_message(f"📊 <b>NUOVA EPOCA: {epoca}</b>")
        
        # Attendi il completamento del processo
        process.wait()
        exit_code = process.returncode
        
        print(f"\n[BOT] Processo terminato con exit code: {exit_code}")
        bot.send_completion_message(script_path, exit_code)
        
        return exit_code
    
    except KeyboardInterrupt:
        print("\n[BOT] Interruzione manuale ricevuta")
        bot.send_message("⚠️ <b>Bot interrotto manualmente</b>")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        return 130
    
    except Exception as e:
        print(f"\n[BOT] Errore durante l'esecuzione: {e}")
        bot.send_message(f"❌ <b>Errore:</b> {str(e)}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Bot Telegram per monitorare l'esecuzione di script Python"
    )
    parser.add_argument(
        "script",
        help="Percorso dello script Python o nome del modulo da eseguire"
    )
    parser.add_argument(
        "script_args",
        nargs="*",
        help="Argomenti da passare allo script"
    )
    parser.add_argument(
        "-m", "--module",
        action="store_true",
        help="Esegui come modulo (python -m) invece di file diretto"
    )
    parser.add_argument(
        "--token",
        default="",
        help="Token del bot Telegram"
    )
    parser.add_argument(
        "--chat-id",
        default="-1003601256539",
        help="ID del canale/chat Telegram"
    )
    
    args = parser.parse_args()
    
    # Crea il bot
    bot = TelegramBot(args.token, args.chat_id)
    
    # Testa la connessione
    print("[BOT] Test connessione Telegram...")
    if bot.send_message("🔧 Test connessione riuscito"):
        print("[BOT] Connessione OK!")
    else:
        print("[BOT] Attenzione: problemi di connessione")
    
    # Monitora il processo
    exit_code = monitor_process(bot, args.script, args.script_args, is_module=args.module)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
