# Motherhood Companion (Preg-chatbot)

### Your 24/7 AI maternal health guide  
A modern, full-stack chatbot for pregnancy and baby care, powered by FastAPI (backend) and React (frontend).


## Features

- **Conversational AI**: Ask questions about pregnancy, baby care, and maternal health.  
- **FAQ Sidebar**: Frequently asked questions for quick access.  
- **Modern UI**: Responsive, professional, and mobile-friendly chat interface.  
- **24/7 Support**: Always online, always ready to help.  
- **Easy Deployment**: Backend on Render, frontend on Vercel.


## Demo

![preg-bot](https://github.com/user-attachments/assets/9ca6663b-d26f-4348-b97d-f775a2cacc3b)



## Getting Started (Local Development)

### 1. Clone the Repository

```bash
git clone https://github.com/sathya-mithra-k/Preg-chatbot.git
cd Preg-chatbot
```
### 2. Backend Setup (FastAPI)
Install dependencies:

```bash
pip install -r requirements_backend.txt
```
Create a .env file with your API keys:

``` Code snippet
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```
Start the backend:

```bash
python backend.py
```
The backend runs at: http://localhost:8000

### 3. Frontend Setup (React)

```bash
cd prebashgnancy-companion-frontend
npm install
```
Create a .env file:

```Code snippet
REACT_APP_API_URL=http://localhost:8000/ask
```

Start the frontend:

```bash
npm start
```
The frontend runs at http://localhost:3000.

### Technologies
- Frontend: React, CSS
- Backend: FastAPI, LangChain, OpenAI, Pinecone
- Hosting: Vercel (frontend), Render (backend)
