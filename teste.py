# -*- coding: utf-8 -*-
"""
teste.py — Reconhecimento facial (cadastro e validação) com dlib + OpenCV
- Sem conexão com Arduino
- Caminhos absolutos para os modelos .dat
- Cadastro sem 'input()' (captura de texto na própria janela)
- Teclas:
    [E] cadastrar (coleta N amostras e salva média)
    [V] ligar/desligar validação
    [D] deletar cadastro do rosto atual (com confirmação)
    [L] listar cadastros
    [Q] sair
"""

import os
import time
import pickle
import numpy as np
import cv2
import dlib

# ---------------------------------------------------------------------
# Configurações e caminhos
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTOR = os.path.join(BASE_DIR, "shape_predictor_5_face_landmarks.dat")
RECOG    = os.path.join(BASE_DIR, "dlib_face_recognition_resnet_model_v1.dat")
DB_FILE  = os.path.join(BASE_DIR, "db.pkl")

THRESH_RECOG = 0.60     # limiar de reconhecimento (quanto menor, mais estrito)
THRESH_DELETE_SUGGEST = 0.70  # limiar só para sugerir nome ao deletar
COOLDOWN  = 3           # segundos entre "reconhecido" consecutivos (efeito visual)
CAMERA_ID = 0           # ID da webcam (0 = padrão)
SAMPLES_N = 5           # N amostras para cadastro (faz média)
SAMPLES_TIMEOUT = 8.0   # tempo máximo (s) para coletar as amostras

WINDOW = "Faces"

# ---------------------------------------------------------------------
# Verificações iniciais
# ---------------------------------------------------------------------
for path, label in [(PREDICTOR, "PREDICTOR"), (RECOG, "RECOG")]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERRO] Arquivo do modelo não encontrado ({label}): {path}\n"
            f"Coloque o .dat correto nessa pasta ou ajuste o caminho."
        )

# Banco de embeddings (nome -> vetor 128D)
db = pickle.load(open(DB_FILE, "rb")) if os.path.exists(DB_FILE) else {}

# ---------------------------------------------------------------------
# Modelos dlib
# ---------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR)
rec = dlib.face_recognition_model_v1(RECOG)

# ---------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------
def draw_text(img, text, org, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def get_name_in_window(frame_bg, prompt="Digite o nome e pressione ENTER (ESC para cancelar)"):
    """
    Captura texto dentro da janela OpenCV (sem input()).
    Enter confirma, ESC cancela, Backspace apaga.
    Aceita A-Z, a-z, 0-9, espaço, _ - .
    Retorna string (pode ser vazia se cancelado).
    """
    name = ""
    valid = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-."
    while True:
        frame = frame_bg.copy()
        draw_text(frame, prompt, (10, 80), 0.7, (0, 255, 255), 2)
        draw_text(frame, f"Nome: {name}", (10, 110), 0.8, (0, 255, 0), 2)
        cv2.imshow(WINDOW, frame)

        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # ESC
            return ""
        elif k in (13, 10):  # Enter
            return name.strip()
        elif k == 8:  # Backspace
            name = name[:-1]
        elif k != 255:
            ch = chr(k)
            if ch in valid and len(name) < 40:
                name += ch

def get_confirmation_in_window(frame_bg, prompt="Pressione ENTER para confirmar, ESC para cancelar"):
    """
    Confirmação simples na janela (ENTER = True, ESC = False).
    """
    while True:
        frame = frame_bg.copy()
        draw_text(frame, prompt, (10, 80), 0.7, (0, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10):  # Enter
            return True
        if k == 27:        # ESC
            return False

def save_db(db_dict, path):
    with open(path, "wb") as f:
        pickle.dump(db_dict, f)

def identify(vec, database, thresh):
    """
    Retorna (nome, dist) do mais próximo no banco.
    Se dist > thresh, retorna ("Desconhecido", dist).
    """
    nome, dist = "Desconhecido", 999.0
    for n, v in database.items():
        d = np.linalg.norm(vec - v)
        if d < dist:
            nome, dist = n, d
    if dist > thresh:
        nome = "Desconhecido"
    return nome, dist

# ---------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("[ERRO] Não foi possível abrir a webcam. Verifique o dispositivo/câmera.")

validando = False
ultimo = 0.0
last_vec = None         # embedding do último frame quando havia 1 rosto
last_rect_count = 0

print("[Controles]  [E]=Cadastrar  [V]=Validar ON/OFF  [D]=Deletar  [L]=Listar  [Q]=Sair")
print(f"[Info] Banco carregado com {len(db)} pessoa(s).")
cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

# ---------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("[Aviso] Frame não capturado da câmera. Encerrando.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = detector(rgb, 0)

    last_rect_count = len(rects)
    last_vec = None

    for r in rects:
        shape = sp(rgb, r)
        chip = dlib.get_face_chip(rgb, shape, size=150)
        vec = np.array(rec.compute_face_descriptor(chip), dtype=np.float32)

        if len(rects) == 1:
            last_vec = vec

        if validando and db:
            nome, dist = identify(vec, db, THRESH_RECOG)
            color = (0, 255, 0) if nome != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (r.left(), r.top()), (r.right(), r.bottom()), color, 2)
            draw_text(frame, f"{nome}", (r.left(), max(25, r.top() - 10)), 0.7, color, 2)

            if nome != "Desconhecido" and (time.time() - ultimo) > COOLDOWN:
                ultimo = time.time()
        else:
            cv2.rectangle(frame, (r.left(), r.top()), (r.right(), r.bottom()), (255, 255, 255), 1)

    draw_text(frame, "[E]=Cadastrar  [V]=Validar ON/OFF  [D]=Deletar  [L]=Listar  [Q]=Sair",
              (10, 25), 0.7, (255, 255, 255), 2)
    draw_text(frame, f"Validacao: {'ON' if validando else 'OFF'}  |  Banco: {len(db)}",
              (10, 55), 0.6, (200, 200, 200), 2)

    if last_rect_count == 0:
        draw_text(frame, "Nenhum rosto detectado", (10, 85), 0.6, (0, 0, 255), 2)
    elif last_rect_count > 1:
        draw_text(frame, "Mostre apenas 1 rosto para cadastrar (E)", (10, 85), 0.6, (0, 165, 255), 2)

    cv2.imshow(WINDOW, frame)

    # ---------------------- teclado ----------------------
    k = cv2.waitKey(1)
    if k != -1:
        try:
            key = chr(k & 0xFF).lower()
        except ValueError:
            key = ""

        if key == 'q':
            print("[Saindo] Encerrando aplicacao.")
            break

        elif key == 'v':
            validando = not validando
            print(f"[Validacao] {'ON' if validando else 'OFF'}")

        elif key == 'l':
            if db:
                print("[Lista] Pessoas cadastradas:", ", ".join(sorted(db.keys())))
            else:
                print("[Lista] Banco vazio.")

        elif key == 'd':
            # Deletar o cadastro do rosto atual (sugere o nome pelo embedding)
            if last_rect_count == 1 and last_vec is not None and db:
                nome, dist = identify(last_vec, db, THRESH_DELETE_SUGGEST)
                if nome == "Desconhecido":
                    print("[Remover] Nao consegui identificar quem remover.")
                else:
                    prompt = f"Remover '{nome}'? ENTER=Sim  ESC=Nao"
                    if get_confirmation_in_window(frame, prompt):
                        db.pop(nome, None)
                        try:
                            save_db(db, DB_FILE)
                            print(f"[Remover] '{nome}' removido. Banco = {len(db)}")
                        except Exception as e:
                            print(f"[ERRO] Falha ao salvar DB: {e}")
                    else:
                        print("[Remover] Cancelado.")
            else:
                print("[Remover] Mostre 1 rosto e tenha pelo menos 1 cadastro no banco.")

        elif key == 'e':
            # Cadastro com múltiplas amostras (média)
            if last_rect_count != 1 or last_vec is None:
                print(f"[Cadastro] Precisa ter exatamente 1 rosto visivel (atual: {last_rect_count}).")
                temp = frame.copy()
                draw_text(temp, "Mostre 1 rosto para cadastrar", (10, 115), 0.7, (0, 0, 255), 2)
                cv2.imshow(WINDOW, temp)
                cv2.waitKey(800)
            else:
                name = get_name_in_window(frame)
                if not name:
                    print("[Cadastro] Cancelado.")
                else:
                    # Coleta N amostras
                    samples = []
                    print(f"[Cadastro] Coletando {SAMPLES_N} amostras de '{name}'. Fique parada(o)...")
                    collected = 0
                    start = time.time()
                    while collected < SAMPLES_N and (time.time() - start) < SAMPLES_TIMEOUT:
                        ok2, f2 = cap.read()
                        if not ok2:
                            break
                        rgb2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                        rects2 = detector(rgb2, 0)
                        if len(rects2) == 1:
                            shape2 = sp(rgb2, rects2[0])
                            chip2 = dlib.get_face_chip(rgb2, shape2, size=150)
                            vec2 = np.array(rec.compute_face_descriptor(chip2), dtype=np.float32)
                            samples.append(vec2)
                            collected += 1
                            draw_text(f2, f"Amostra {collected}/{SAMPLES_N}", (10, 140), 0.7, (0, 255, 0), 2)
                        else:
                            draw_text(f2, "Precisamos de 1 rosto visivel", (10, 140), 0.7, (0, 165, 255), 2)
                        cv2.imshow(WINDOW, f2)
                        cv2.waitKey(60)

                    if len(samples) > 0:
                        emb = np.mean(samples, axis=0).astype(np.float32)
                        db[name] = emb
                        try:
                            save_db(db, DB_FILE)
                            print(f"[Cadastro] {name} salvo(a) com {len(samples)} amostras. Banco = {len(db)}")
                        except Exception as e:
                            print(f"[ERRO] Falha ao salvar DB: {e}")
                    else:
                        print("[Cadastro] Nao foi possivel coletar amostras.")

# ---------------------------------------------------------------------
# Finalização
# ---------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
print("[OK] Recursos liberados.")
