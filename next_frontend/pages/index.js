import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';

const STORAGE_KEY = 'stelligence_chats_v3';
const SUGGESTIONS = [
  'Who is in the network?',
  'Find path from Por to Anutin',
  'Who is the most connected person?'
];

function loadChats() {
  if (typeof window === 'undefined') return [];
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch { return []; }
}

function saveChats(chats) {
  if (typeof window !== 'undefined') {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(chats)); } catch {}
  }
}

export default function Home() {
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingTime, setThinkingTime] = useState(0);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const thinkingTimerRef = useRef(null);

  useEffect(() => {
    const loaded = loadChats();
    setChats(loaded);
    if (loaded.length > 0) {
      setCurrentChatId(loaded[0].id);
      setMessages(loaded[0].messages || []);
    }
  }, []);

  useEffect(() => { if (chats.length > 0) saveChats(chats); }, [chats]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isThinking]);
  useEffect(() => { setTimeout(() => inputRef.current?.focus(), 100); }, [currentChatId, isThinking]);

  useEffect(() => {
    if (isThinking) {
      setThinkingTime(0);
      thinkingTimerRef.current = setInterval(() => setThinkingTime(t => t + 0.1), 100);
    } else if (thinkingTimerRef.current) {
      clearInterval(thinkingTimerRef.current);
      thinkingTimerRef.current = null;
    }
    return () => { if (thinkingTimerRef.current) clearInterval(thinkingTimerRef.current); };
  }, [isThinking]);

  const createNewChat = () => {
    const id = Date.now().toString();
    setChats(prev => [{ id, name: 'New chat', messages: [] }, ...prev]);
    setCurrentChatId(id);
    setMessages([]);
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  const selectChat = (id) => {
    if (isThinking) return;
    const chat = chats.find(c => c.id === id);
    if (chat) {
      setCurrentChatId(id);
      setMessages(chat.messages || []);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  };

  const deleteChat = (e, id) => {
    e.stopPropagation();
    const remaining = chats.filter(c => c.id !== id);
    setChats(remaining);
    if (currentChatId === id) {
      if (remaining.length > 0) {
        setCurrentChatId(remaining[0].id);
        setMessages(remaining[0].messages || []);
      } else {
        setCurrentChatId(null);
        setMessages([]);
      }
    }
  };

  const copyText = (text) => navigator.clipboard.writeText(text);

  const sendMessage = async (text) => {
    const trimmed = (text || '').trim();
    if (!trimmed || isThinking) return;

    setInput('');
    setIsThinking(true);

    let chatId = currentChatId;
    let updatedMessages = [...messages];

    if (!chatId) {
      chatId = Date.now().toString();
      setChats(prev => [{ id: chatId, name: trimmed.slice(0, 30), messages: [] }, ...prev]);
      setCurrentChatId(chatId);
      updatedMessages = [];
    }

    const userMsg = { role: 'user', content: trimmed };
    updatedMessages = [...updatedMessages, userMsg];
    setMessages(updatedMessages);
    setChats(prev => prev.map(c => c.id === chatId ? { ...c, messages: updatedMessages, name: trimmed.slice(0, 30) } : c));

    const startTime = Date.now();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: trimmed })
      });
      const data = await res.json();
      const assistantMsg = {
        role: 'assistant',
        content: data.answer || data.error || 'No response received.',
        duration: ((Date.now() - startTime) / 1000).toFixed(1),
        prompt: trimmed
      };
      updatedMessages = [...updatedMessages, assistantMsg];
    } catch (err) {
      updatedMessages = [...updatedMessages, { role: 'assistant', content: 'Error: ' + err.message, prompt: trimmed }];
    }

    setMessages(updatedMessages);
    setChats(prev => prev.map(c => c.id === chatId ? { ...c, messages: updatedMessages } : c));
    setIsThinking(false);
  };

  const regenerate = async (prompt) => {
    if (!prompt || isThinking) return;
    setIsThinking(true);
    const startTime = Date.now();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt })
      });
      const data = await res.json();
      const newMsg = {
        role: 'assistant',
        content: data.answer || data.error || 'No response received.',
        duration: ((Date.now() - startTime) / 1000).toFixed(1),
        prompt
      };
      const updated = [...messages];
      const lastIdx = updated.map(m => m.role).lastIndexOf('assistant');
      if (lastIdx >= 0) updated[lastIdx] = newMsg;
      setMessages(updated);
      setChats(prev => prev.map(c => c.id === currentChatId ? { ...c, messages: updated } : c));
    } catch {}
    setIsThinking(false);
  };

  const handleSubmit = (e) => { e.preventDefault(); sendMessage(input); };
  const showWelcome = messages.length === 0 && !isThinking;

  return (
    <div className="container">
      <Head><title>STelligence Network Agent</title></Head>
      <div className="sidebar">
        <button className="new-chat-btn" onClick={createNewChat}>+ New chat</button>
        <div className="chat-list">
          {chats.map(chat => (
            <div key={chat.id} className={chat.id === currentChatId ? 'chat-item active' : 'chat-item'} onClick={() => selectChat(chat.id)}>
              <span className="chat-title">{chat.name}</span>
              <button className="delete-btn" onClick={(e) => deleteChat(e, chat.id)}>x</button>
            </div>
          ))}
        </div>
        <div className="sidebar-footer">STelligence Network Agent</div>
      </div>
      <div className="main">
        <div className="messages-container">
          {showWelcome ? (
            <div className="welcome">
              <h1>How can I help you today?</h1>
              <div className="suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <button key={i} className="suggestion-btn" onClick={() => sendMessage(s)}>{s}</button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg, idx) => (
                <div key={idx} className={'message-row ' + msg.role}>
                  <div className={'bubble ' + msg.role}>
                    <div className="msg-text">{msg.content}</div>
                    {msg.role === 'assistant' && (
                      <div className="actions">
                        {msg.duration && <span className="dur">{msg.duration}s</span>}
                        <button onClick={() => copyText(msg.content)}>Copy</button>
                        <button onClick={() => regenerate(msg.prompt)}>Regenerate</button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isThinking && (
                <div className="message-row assistant">
                  <div className="bubble assistant">
                    <div className="thinking">
                      <span className="dots"></span>
                      <span>Thinking... {thinkingTime.toFixed(1)}s</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
        <div className="input-area">
          <form onSubmit={handleSubmit} className="input-form">
            <input ref={inputRef} type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Message STelligence..." disabled={isThinking} autoFocus />
            <button type="submit" disabled={!input.trim() || isThinking} className="send-btn">Send</button>
          </form>
          <p className="disclaimer">Powered by Neo4j Graph Database and Local LLM</p>
        </div>
      </div>
      <style jsx>{`
        .container { display: flex; height: 100vh; background: #212121; color: #ececf1; font-family: 'Segoe UI', sans-serif; }
        .sidebar { width: 260px; background: #171717; display: flex; flex-direction: column; padding: 8px; }
        .new-chat-btn { display: flex; align-items: center; gap: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; background: #212121; color: #fff; font-size: 14px; cursor: pointer; width: 100%; }
        .new-chat-btn:hover { background: #2a2a2a; border-color: rgba(255,255,255,0.25); }
        .chat-list { flex: 1; overflow-y: auto; margin-top: 8px; }
        .chat-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-radius: 6px; cursor: pointer; margin-bottom: 2px; }
        .chat-item:hover, .chat-item.active { background: #212121; }
        .chat-title { font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
        .delete-btn { background: none; border: none; color: #8e8ea0; cursor: pointer; padding: 4px 8px; opacity: 0; font-size: 14px; }
        .chat-item:hover .delete-btn { opacity: 1; }
        .delete-btn:hover { color: #ef4444; }
        .sidebar-footer { padding: 12px; border-top: 1px solid rgba(255,255,255,0.1); margin-top: 8px; font-size: 12px; color: #8e8ea0; }
        .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; background: #212121; }
        .messages-container { flex: 1; overflow-y: auto; }
        .welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; padding: 20px; }
        .welcome h1 { font-size: 32px; font-weight: 600; margin: 0 0 32px 0; color: #fff; }
        .suggestions { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; max-width: 600px; }
        .suggestion-btn { padding: 12px 16px; background: #171717; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; color: #ececf1; font-size: 14px; cursor: pointer; transition: all 0.2s; }
        .suggestion-btn:hover { background: #212121; border-color: #10a37f; }
        .messages { max-width: 900px; margin: 0 auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .message-row { display: flex; width: 100%; }
        .message-row.user { justify-content: flex-end; }
        .message-row.assistant { justify-content: flex-start; }
        .bubble { max-width: 70%; padding: 12px 16px; border-radius: 18px; }
        .bubble.user { background: #10a37f; color: #fff; border-bottom-right-radius: 4px; }
        .bubble.assistant { background: #171717; color: #ececf1; border-bottom-left-radius: 4px; }
        .msg-text { white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; font-size: 15px; }
        .actions { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); }
        .actions button { background: none; border: 1px solid rgba(255,255,255,0.2); color: #8e8ea0; cursor: pointer; padding: 4px 10px; border-radius: 4px; font-size: 12px; }
        .actions button:hover { color: #ececf1; background: rgba(255,255,255,0.1); }
        .dur { font-size: 12px; color: #8e8ea0; }
        .thinking { display: flex; align-items: center; gap: 12px; color: #8e8ea0; font-size: 14px; }
        .dots { display: inline-block; width: 40px; }
        .dots::after { content: ''; display: inline-block; animation: dots 1.5s steps(4, end) infinite; }
        @keyframes dots { 0% { content: ''; } 25% { content: '.'; } 50% { content: '..'; } 75% { content: '...'; } }
        .input-area { padding: 16px 24px 24px; background: #212121; }
        .input-form { display: flex; align-items: center; max-width: 768px; margin: 0 auto; background: #171717; border: 1px solid rgba(255,255,255,0.15); border-radius: 24px; padding: 8px 16px; }
        .input-form:focus-within { border-color: #10a37f; }
        .input-form input { flex: 1; background: transparent; border: none; outline: none; color: #fff; font-size: 16px; padding: 8px 0; }
        .input-form input::placeholder { color: #8e8ea0; }
        .input-form input:disabled { opacity: 0.5; }
        .send-btn { background: #10a37f; border: none; color: #fff; cursor: pointer; padding: 8px 16px; border-radius: 20px; font-size: 14px; }
        .send-btn:hover:not(:disabled) { background: #0d8a6a; }
        .send-btn:disabled { opacity: 0.3; cursor: not-allowed; }
        .disclaimer { text-align: center; font-size: 12px; color: #8e8ea0; margin: 12px 0 0 0; }
      `}</style>
    </div>
  );
}
