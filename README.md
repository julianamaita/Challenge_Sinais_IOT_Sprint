# 🔍 Reconhecimento Facial com Dlib e OpenCV

Alunos
Juliana Villalpando Maita - 99224 
João Victor Dos Santos Morais - 550453 
Luana Cabezaolias Miguel - 99320 
Lucca Vilaça Okubo - 551538 
Pedro Henrique Pontes Farath - 98608  

## 🎯 Objetivo
Este projeto demonstra uma aplicação local de **reconhecimento e identificação facial** utilizando **Python, Dlib e OpenCV**.  
Não há necessidade de conexão com internet ou banco de dados externo: todos os dados ficam salvos em um arquivo local (`db.pkl`).  

O sistema permite:
- Cadastrar múltiplos usuários (com várias amostras para cada pessoa).
- Validar rostos em tempo real com feedback visual.
- Listar e excluir cadastros diretamente pela interface.

---

## 📦 Dependências

* Python 3.8+
* OpenCV
* Dlib
* NumPy


## ⌨️ Controles

* **E** → Cadastrar novo rosto (quando houver **1 rosto detectado**).
  O sistema coleta múltiplas amostras e gera uma média para maior precisão.
* **V** → Ativar/desativar validação em tempo real.
* **D** → Deletar o rosto atual do banco de dados (confirmação na janela).
* **L** → Listar todos os rostos cadastrados no console.
* **Q** → Sair do programa.

---

## 📂 Estrutura do Projeto

```
├── teste.py                  # Script principal
├── db.pkl                    # Banco de rostos cadastrados (gerado automaticamente)
├── shape_predictor_5_face_landmarks.dat
├── dlib_face_recognition_resnet_model_v1.dat
├── requirements.txt
└── README.md
```

---

## 📜 Nota Ética

O reconhecimento facial é uma tecnologia poderosa, mas deve ser utilizado de forma **responsável e ética**.

Neste projeto:

* O armazenamento é **local** e os dados faciais são gravados apenas com o consentimento do usuário.
* O sistema **não coleta nem compartilha informações** com terceiros.
* Este código foi desenvolvido para **fins educacionais e acadêmicos**, não devendo ser usado em aplicações que possam violar privacidade ou direitos individuais.
