import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './components/ChatMessage';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/ask';

const FAQS = [
  'How is my baby developing this week?',
  'What foods should I eat?',
  "I\'m experiencing morning sickness",
  'What are the signs of labor?',
  'How much weight should I gain during pregnancy?',
  'What prenatal vitamins do I need?'
];

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function App() {
  const [messages, setMessages] = useState([
    {
      text: "Hello! I'm your pregnancy companion. I'm here to support you 24/7 with personalized guidance for your journey. How are you feeling today?",
      time: getCurrentTime(),
      isUser: false
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async (text, isUser = true) => {
    if (!text.trim()) return;
    const userMsg = { text, time: getCurrentTime(), isUser: true };
    setMessages(msgs => [...msgs, userMsg]);
    setLoading(true);
    setInput('');
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text, include_context: false })
      });
      const data = await res.json();
      setMessages(msgs => [
        ...msgs,
        { text: data.answer || 'Sorry, I could not get a response.', time: getCurrentTime(), isUser: false }
      ]);
    } catch (err) {
      setMessages(msgs => [
        ...msgs,
        { text: 'Sorry, there was a problem connecting to the server.', time: getCurrentTime(), isUser: false }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleInput = e => setInput(e.target.value);
  const handleSend = () => sendMessage(input);
  const handleKeyDown = e => { if (e.key === 'Enter') handleSend(); };
  const handleFAQClick = q => sendMessage(q);

  return (
    <div className="app-main-layout">
      <div className="chat-app-container">
        <header className="chat-header">
          <div className="chat-header-icon"> <span role="img" aria-label="heart">💗</span> </div>
          <div>
            <div className="chat-header-title">Pregnancy Companion</div>
            <div className="chat-header-subtitle">Your 24/7 AI maternal health guide</div>
          </div>
          <div className="chat-header-status">Online 24/7</div>
        </header>
        <main className="chat-main">
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} message={msg} isUser={msg.isUser} />
            ))}
            <div ref={chatEndRef} />
          </div>
        </main>
        <footer className="chat-footer">
          <input
            className="chat-input"
            type="text"
            placeholder="Type your question or concern..."
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button className="chat-send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
            <span role="img" aria-label="send">{loading ? '⏳' : '➤'}</span>
          </button>
          <button className="chat-mic-btn" disabled>
            <span role="img" aria-label="mic">🎤</span>
          </button>
        </footer>
      </div>
      <aside className="faq-section">
        <div className="faq-title">Frequently Asked Questions</div>
        <div className="faq-list">
          {FAQS.map((q, idx) => (
            <button key={idx} className="faq-item" onClick={() => handleFAQClick(q)}>{q}</button>
          ))}
        </div>
      </aside>
    </div>
  );
}

export default App;
