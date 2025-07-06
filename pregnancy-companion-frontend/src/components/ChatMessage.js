import React from 'react';
import './ChatMessage.css';

export default function ChatMessage({ message, isUser }) {
  return (
    <div className={`chat-message-row ${isUser ? 'user' : 'bot'}`}>  
      {!isUser && (
        <div className="chat-avatar bot-avatar">
          <span role="img" aria-label="bot">🩺</span>
        </div>
      )}
      <div className={`chat-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>  
        <div className="chat-message-text">{message.text}</div>
        <div className="chat-message-time">{message.time}</div>
      </div>
      {isUser && (
        <div className="chat-avatar user-avatar">
          <span role="img" aria-label="user">🧑</span>
        </div>
      )}
    </div>
  );
} 