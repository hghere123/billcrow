import { useState } from 'react'
import './index.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hello! I am your AI assistant. Ask me anything about the documents!' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.text })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Server error');
      }
      
      const assistantMessage = {
        role: 'assistant',
        text: data.answer,
        contexts: data.contexts
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'assistant', text: `Error: ${error.message} (Is the backend running and indexed?)` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Mini RAG Chat</h1>
      </header>
      
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-wrapper ${msg.role}`}>
            <div className="message-content">
              <p>{msg.text}</p>
              
              {msg.contexts && msg.contexts.length > 0 && (
                <details className="context-details">
                  <summary>View Sources ({msg.contexts.length})</summary>
                  <div className="context-list">
                    {msg.contexts.map((ctx, i) => (
                      <div key={i} className="context-item">
                        <span className="context-source">{ctx.source}</span>
                        <p className="context-text">{ctx.text}</p>
                        <span className="context-score">Relevance Match: {ctx.score.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message-wrapper assistant">
            <div className="message-content loading">Generating response...</div>
          </div>
        )}
      </div>

      <form className="chat-input-form" onSubmit={sendMessage}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>Send</button>
      </form>
    </div>
  )
}

export default App
