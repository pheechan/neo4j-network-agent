import React, {useState, useEffect, useRef} from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function MessageCard({m}){
  return (
    <div className={"message-card " + (m.role === 'user' ? 'user' : 'assistant')}>
      <div className="meta">
        <span className="role">{m.role === 'user' ? 'You' : 'Assistant'}</span>
        <span className="time">{m.time || ''}</span>
      </div>
      <div className="content"><pre>{m.text}</pre></div>
      <div className="actions">
        {/* placeholder for message actions: copy, pin, edit */}
        <button className="small">Copy</button>
      </div>
    </div>
  )
}

function makeId(){ return Math.random().toString(36).slice(2,9) }

export default function App(){
  // Conversations stored in localStorage for persistence
  const [threads, setThreads] = useState(()=>{
    try{
      const raw = localStorage.getItem('threads')
      return raw ? JSON.parse(raw) : [{id: 't1', title: 'New Chat', messages: []}]
    }catch(e){ return [{id: 't1', title: 'New Chat', messages: []}] }
  })
  const [currentThread, setCurrentThread] = useState(threads[0]?.id)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef()

  useEffect(()=>{ localStorage.setItem('threads', JSON.stringify(threads)) }, [threads])
  useEffect(()=>{ bottomRef.current?.scrollIntoView({behavior:'smooth'}) }, [threads, currentThread])

  const thread = threads.find(t=>t.id === currentThread) || threads[0]

  function newThread(){
    const id = makeId()
    const t = {id, title: 'New Chat', messages: []}
    setThreads(s=>[t, ...s])
    setCurrentThread(id)
  }

  function selectThread(id){ setCurrentThread(id) }

  function appendMessage(role, text){
    const msg = {id: makeId(), role, text, time: new Date().toLocaleString()}
    setThreads(prev => prev.map(t => t.id === currentThread ? {...t, messages: [...t.messages, msg]} : t))
    return msg
  }

  function updateLastAssistantText(text){
    setThreads(prev => prev.map(t => {
      if(t.id !== currentThread) return t
      const msgs = [...t.messages]
      // find last assistant message
      for(let i=msgs.length-1;i>=0;i--){ if(msgs[i].role === 'assistant'){ msgs[i] = {...msgs[i], text}; break } }
      return {...t, messages: msgs}
    }))
  }

  function send(){
    if(!input.trim()) return
    appendMessage('user', input)
    setInput('')
    setLoading(true)

    // create empty assistant message that we'll fill via SSE
    appendMessage('assistant', '')

    const url = `${API_URL}/api/stream-chat?message=${encodeURIComponent(input)}`
    const es = new EventSource(url)
    let buffer = ''

    es.onmessage = (e) => {
      buffer += e.data
      updateLastAssistantText(buffer + '▌')
    }

    es.onerror = (e) => {
      // close and finalize
      es.close()
      updateLastAssistantText(buffer)
      setLoading(false)
    }
  }

  // Pinned suggestions (static for now)
  const suggestions = [
    'ใครคือสมาชิกของเครือข่าย Santisook?',
    'หาเส้นทางระหว่าง "อนุทิน" และ "พี่โด่ง"',
    'ตำแหน่งของ อนุทิน ชาญวีรกูล'
  ]

  return (
    <div className="app-root">
      <aside className="sidebar">
        <div className="brand">STelligence</div>
        <div className="thread-list">
          <div className="threads-header">
            <strong>Conversations</strong>
            <button onClick={newThread} className="small">+ New</button>
          </div>
          {threads.map(t=> (
            <div key={t.id} className={"thread-item " + (t.id === currentThread ? 'active':'')} onClick={()=>selectThread(t.id)}>
              <div className="title">{t.title}</div>
              <div className="meta">{t.messages.length} msgs</div>
            </div>
          ))}
        </div>

        <div className="pinned">
          <strong>Pinned</strong>
          {suggestions.map((s,i)=>(<button key={i} className="chip" onClick={()=>{ setInput(s); }} >{s}</button>))}
        </div>
      </aside>

      <main className="main">
        <div className="chat-header">{thread?.title || 'Chat'}</div>
        <div className="chat-window">
          {thread?.messages?.map(m=> <MessageCard key={m.id} m={m} />)}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <textarea value={input} onChange={e=>setInput(e.target.value)} placeholder="Ask me about the network..." />
          <div className="controls">
            <button onClick={send} disabled={loading} className="send">{loading ? 'Sending...' : 'Send'}</button>
          </div>
        </div>
      </main>
    </div>
  )
}
