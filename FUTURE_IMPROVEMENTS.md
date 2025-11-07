# 🚀 Additional Improvements for Neo4j Network Agent

## Recently Fixed (v2.1.0)
✅ **Cache Control Issues**
- Added "💾 Enable caching" checkbox in settings
- Added "🗑️ Clear all caches" button
- Regenerate button now bypasses cache for fresh answers
- New chat properly gets fresh results (not cached)

---

## 🎯 Recommended Next Improvements

### 1. **Smart Query Auto-complete** 🔍
**What:** Suggest queries as user types (like Google)

**How to implement:**
```python
# Store popular queries
popular_queries = [
    "ใครคือนายกรัฐมนตรี?",
    "รัฐมนตรีว่าการกระทรวงการคลัง?",
    "Stelligence network มีใครบ้าง?",
]

# In chat input
user_input = st.chat_input("Send a message...")
if user_input and len(user_input) > 3:
    matches = [q for q in popular_queries if user_input.lower() in q.lower()]
    if matches:
        st.caption("💡 Did you mean: " + " | ".join(matches[:3]))
```

**Benefits:**
- Faster query entry
- Discover what system can answer
- Reduce typos

---

### 2. **Query History Search** 📚
**What:** Search through past conversations

**How to implement:**
```python
# In sidebar
with st.expander("🔍 Search History"):
    search_term = st.text_input("Search past conversations...")
    if search_term:
        results = []
        for tid, thread in st.session_state.threads.items():
            for msg in thread['messages']:
                if search_term.lower() in msg['content'].lower():
                    results.append((tid, msg))
        
        for tid, msg in results[:10]:
            if st.button(f"📝 {msg['content'][:50]}...", key=f"search_{tid}_{msg['time']}"):
                st.session_state.current_thread = tid
                st.rerun()
```

**Benefits:**
- Find previous answers quickly
- Avoid asking same question
- Build knowledge base

---

### 3. **Conversation Export** 💾
**What:** Export chat to PDF, Word, or Markdown

**How to implement:**
```python
def export_conversation(thread_id, format='markdown'):
    thread = st.session_state.threads[thread_id]
    
    if format == 'markdown':
        md = f"# {thread['title']}\n\n"
        md += f"**Created:** {thread['created_at']}\n\n"
        for msg in thread['messages']:
            role = "👤 User" if msg['role'] == 'user' else "🔮 Assistant"
            md += f"## {role} ({msg['time']})\n\n"
            md += f"{msg['content']}\n\n---\n\n"
        return md
    
    elif format == 'json':
        return json.dumps(thread, ensure_ascii=False, indent=2)

# In sidebar
if st.button("📥 Export Current Chat"):
    md_content = export_conversation(st.session_state.current_thread)
    st.download_button(
        "Download as Markdown",
        md_content,
        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )
```

**Benefits:**
- Save important conversations
- Share findings with team
- Create reports from chats

---

### 4. **Voice Input** 🎤
**What:** Ask questions by speaking (Thai & English)

**How to implement:**
```python
# Install: pip install streamlit-webrtc SpeechRecognition
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr

# In chat interface
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.chat_input("Send a message...")
with col2:
    if st.button("🎤", help="Voice input"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("🎤 Listening...")
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio, language='th-TH')
                st.session_state['voice_input'] = text
                st.rerun()
            except:
                st.error("Couldn't understand audio")

# Process voice input
if 'voice_input' in st.session_state:
    user_input = st.session_state.pop('voice_input')
```

**Benefits:**
- Hands-free operation
- Faster for long queries
- Accessibility feature

---

### 5. **Answer Rating & Feedback** ⭐
**What:** Let users rate answers to improve system

**How to implement:**
```python
# After each answer
col1, col2, col3, col4 = st.columns([1, 1, 1, 9])
with col1:
    if st.button("👍", key=f"like_{idx}"):
        log_feedback(thread_id, idx, rating=1, feedback="helpful")
        st.success("Thanks!")
with col2:
    if st.button("👎", key=f"dislike_{idx}"):
        log_feedback(thread_id, idx, rating=-1, feedback="not_helpful")
with col3:
    if st.button("💬", key=f"comment_{idx}"):
        st.session_state['feedback_mode'] = (thread_id, idx)

def log_feedback(thread_id, msg_idx, rating, feedback):
    with open('feedback.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'thread_id': thread_id,
            'message_idx': msg_idx,
            'rating': rating,
            'feedback': feedback,
            'query': st.session_state.threads[thread_id]['messages'][msg_idx-1]['content']
        }, ensure_ascii=False) + '\n')
```

**Benefits:**
- Identify bad answers
- Improve prompts based on feedback
- Track user satisfaction

---

### 6. **Visual Relationship Graph** 🕸️
**What:** Show connections visually (interactive graph)

**How to implement:**
```python
# Install: pip install pyvis networkx
from pyvis.network import Network
import networkx as nx

def visualize_relationships(person_name):
    # Query Neo4j for person + 1-hop connections
    query = """
    MATCH (p:Person {name: $name})-[r]-(connected)
    RETURN p, r, connected
    LIMIT 50
    """
    
    driver = get_driver()
    with driver.session(database=NEO4J_DB) as session:
        result = session.run(query, name=person_name)
        
        net = Network(height="600px", width="100%", bgcolor="#1e1e1e", font_color="white")
        net.barnes_hut()
        
        # Add nodes and edges
        for record in result:
            p = record['p']
            connected = record['connected']
            r = record['r']
            
            net.add_node(p['name'], label=p['name'], color='#ff6b6b')
            net.add_node(connected.get('name', 'Unknown'), label=connected.get('name', 'Unknown'), color='#4ecdc4')
            net.add_edge(p['name'], connected.get('name', 'Unknown'), title=type(r).__name__)
        
        net.save_graph("temp_graph.html")
        with open("temp_graph.html", 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=600)

# In answer section
if st.button("🕸️ Visualize Relationships"):
    visualize_relationships(person_name)
```

**Benefits:**
- Understand complex networks visually
- Discover hidden connections
- Interactive exploration

---

### 7. **Bulk Question Mode** 📋
**What:** Ask multiple questions at once and get all answers

**How to implement:**
```python
# Add bulk mode toggle
bulk_mode = st.checkbox("📋 Bulk Question Mode")

if bulk_mode:
    st.write("Enter questions (one per line):")
    questions = st.text_area("Questions", height=200)
    
    if st.button("🚀 Process All"):
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]
        
        results = []
        progress_bar = st.progress(0)
        for i, q in enumerate(question_list):
            st.caption(f"Processing {i+1}/{len(question_list)}: {q}")
            
            # Get answer (use existing pipeline)
            answer = process_single_question(q)
            results.append({'question': q, 'answer': answer})
            
            progress_bar.progress((i+1) / len(question_list))
        
        # Display all results
        for r in results:
            with st.expander(f"Q: {r['question'][:80]}..."):
                st.markdown(r['answer'])
        
        # Export option
        export_bulk_results(results)
```

**Benefits:**
- Research mode (many questions at once)
- Batch processing for reports
- Save time on repetitive queries

---

### 8. **Smart Context Expansion** 🎯
**What:** Automatically fetch more context if answer is uncertain

**How to implement:**
```python
def analyze_answer_confidence(answer, context):
    """Check if answer seems uncertain"""
    uncertain_phrases = [
        "ไม่แน่ใจ", "อาจจะ", "ไม่มีข้อมูล", "ไม่ทราบ",
        "not sure", "may be", "no information"
    ]
    
    is_uncertain = any(phrase in answer.lower() for phrase in uncertain_phrases)
    return is_uncertain

# In main pipeline
if analyze_answer_confidence(answer, ctx):
    st.warning("⚠️ Answer seems uncertain. Expanding search...")
    
    # Double the search results
    expanded_results = cached_vector_search(
        process_message,
        top_k_per_index=60,  # 2x normal
        _cache_bypass=time.time()
    )
    
    ctx_expanded = build_context(expanded_results)
    
    # Re-generate with more context
    answer = ask_openrouter_requests(
        f"Context: {ctx_expanded}\n\nQuestion: {process_message}",
        max_tokens=2048
    )
```

**Benefits:**
- Better answers when initial search insufficient
- Automatic quality improvement
- Reduced "no information" responses

---

### 9. **Scheduled Reports** 📊
**What:** Auto-generate daily/weekly summaries

**How to implement:**
```python
# Create background task
def generate_daily_report():
    """Run this daily via cron or scheduler"""
    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_queries': 0,
        'unique_users': set(),
        'popular_queries': {},
        'error_count': 0
    }
    
    # Parse analytics
    with open('query_analytics.jsonl', 'r') as f:
        for line in f:
            log = json.loads(line)
            report['total_queries'] += 1
            
            query = log['query']
            report['popular_queries'][query] = report['popular_queries'].get(query, 0) + 1
            
            if not log['success']:
                report['error_count'] += 1
    
    # Top 10 queries
    top_queries = sorted(report['popular_queries'].items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Send report via email or save
    markdown_report = f"""
    # Daily Report - {report['date']}
    
    ## Summary
    - Total Queries: {report['total_queries']}
    - Error Rate: {report['error_count']/report['total_queries']*100:.1f}%
    
    ## Top 10 Queries
    {chr(10).join([f"{i+1}. {q} ({count} times)" for i, (q, count) in enumerate(top_queries)])}
    """
    
    with open(f"reports/daily_{report['date']}.md", 'w', encoding='utf-8') as f:
        f.write(markdown_report)

# Add to sidebar
if st.button("📊 Generate Report"):
    generate_daily_report()
    st.success("Report generated!")
```

**Benefits:**
- Track usage trends
- Identify popular topics
- Monitor system health

---

### 10. **Multi-language Support** 🌍
**What:** Support English, Thai, and other languages automatically

**How to implement:**
```python
# Install: pip install langdetect
from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'th'  # default to Thai

# Adapt system prompt based on language
query_lang = detect_language(process_message)

if query_lang == 'en':
    system_prompt = """You are an expert...[English version]"""
elif query_lang == 'th':
    system_prompt = """คุณเป็นผู้ช่วย...[Thai version]"""

# Use bilingual LLM for translation if needed
if query_lang != 'th' and ctx:  # Context is in Thai
    st.caption("🌐 Translating context to match your language...")
```

**Benefits:**
- Serve international users
- Auto-detect and adapt
- Bilingual knowledge base

---

## 📊 Priority Ranking

| Feature | Impact | Effort | Priority | Users Will Love |
|---------|--------|--------|----------|-----------------|
| Answer Rating | High | Low | ⭐⭐⭐⭐⭐ | Yes! Feels heard |
| Cache Control (✅ Done) | High | Low | ⭐⭐⭐⭐⭐ | Yes! |
| Export Conversations | Medium | Low | ⭐⭐⭐⭐ | Very useful |
| Query Auto-complete | Medium | Medium | ⭐⭐⭐⭐ | Nice to have |
| Voice Input | High | Medium | ⭐⭐⭐ | Cool factor |
| Visual Graph | High | High | ⭐⭐⭐ | Impressive |
| History Search | Medium | Low | ⭐⭐⭐ | Useful |
| Bulk Questions | Low | Medium | ⭐⭐ | Power users |
| Smart Expansion | Medium | High | ⭐⭐ | Behind scenes |
| Scheduled Reports | Low | High | ⭐ | Admin tool |

---

## 🚀 Quick Wins (Do These First!)

1. **Answer Rating** (30 min) - Immediate feedback loop
2. **Export Conversations** (1 hour) - Very requested feature
3. **History Search** (1 hour) - Productivity boost
4. **Query Auto-complete** (2 hours) - Better UX

---

## 🎯 Game Changers (Worth the Effort!)

1. **Visual Relationship Graph** - Makes complex data clear
2. **Voice Input** - Accessibility + coolness
3. **Smart Context Expansion** - Better answer quality

---

Would you like me to implement any of these? I recommend starting with **Answer Rating** - it's quick and gives you valuable feedback!
