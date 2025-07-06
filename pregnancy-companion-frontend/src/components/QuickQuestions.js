import React from 'react';
import './QuickQuestions.css';

export default function QuickQuestions({ questions, onSelect }) {
  return (
    <div className="quick-questions">
      {questions.map((q, idx) => (
        <button key={idx} className="quick-question-btn" onClick={() => onSelect(q)}>
          {q}
        </button>
      ))}
    </div>
  );
} 