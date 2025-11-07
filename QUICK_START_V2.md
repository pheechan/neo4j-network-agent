# 🚀 Quick Start Guide - Version 2.0.0 Features

## New Features You Can Use Right Now!

### 1. 🌊 **Enable Streaming Responses**
**What it does:** See text appear word-by-word (like ChatGPT)

**How to enable:**
1. Open sidebar (left side)
2. Click "⚙️ Settings"
3. Check "🌊 Streaming responses"
4. Ask any question - watch it stream!

**When to use:**
- ✅ Long detailed answers (better UX)
- ❌ Short quick answers (regular is faster)

---

### 2. 📊 **View Analytics**
**What it shows:** Success rate, response time, total queries

**Where to find:**
1. Open sidebar
2. Click "⚙️ Settings"
3. Scroll to "📊 Analytics"

**Metrics:**
- Total queries: How many questions asked
- Success rate: % that got good answers
- Avg response time: How fast responses come

---

### 3. 💡 **Use Follow-up Questions**
**What it does:** Suggests related questions automatically

**How it works:**
1. Ask any question (e.g., "ใครคือนายกรัฐมนตรี?")
2. Get answer
3. Scroll down to see "💡 คำถามที่คุณอาจสนใจ:"
4. Click/copy any suggested question

**Example:**
```
Q: "อนุทิน ชาญวีรกูล ตำแหน่งอะไร?"

A: "อนุทิน ชาญวีรกูล ดำรงตำแหน่ง นายกรัฐมนตรี..."

💡 คำถามที่คุณอาจสนใจ:
• อนุทินมีความสัมพันธ์กับใครบ้างในรัฐบาล?
• กระทรวงมหาดไทยมีหน้าที่อะไรบ้าง?
• รัฐมนตรีช่วยว่าการกระทรวงมหาดไทยคือใคร?
```

---

### 4. 🔗 **Find Connection Paths**
**What it does:** Find how two people are connected

**How to ask:**
- "X รู้จัก Y ผ่านใครบ้าง?"
- "X เชื่อมกับ Y อย่างไร?"
- "ความสัมพันธ์ระหว่าง X กับ Y"

**Example:**
```
Q: "อนุทินรู้จักจุรินทร์ผ่านใครบ้าง?"

System will:
1. Detect relationship query
2. Find shortest path (max 3 hops)
3. Show connection chain
```

---

### 5. ⚡ **Faster Repeat Queries**
**What it does:** Cache results for 1 hour

**How it works:**
- First time: Search database + call LLM (3-5s)
- Second time: Return cached result (0.1-0.5s)
- Cache expires: After 1 hour

**When you'll notice:**
- Same question twice → instant answer
- Similar questions → faster search
- Browsing history → quick loads

**Tip:** If data changed, wait 1 hour or use "🔄 Regenerate"

---

### 6. 🛡️ **Auto-Retry on Errors**
**What it does:** Retry failed requests automatically

**Handles:**
- 429 (rate limit): Waits 2s → 4s → 8s then retries
- 5xx (server error): Retries up to 3 times

**You'll see:**
```
⏳ Rate limited. Retrying in 2s... (Attempt 1/3)
⏳ Rate limited. Retrying in 4s... (Attempt 2/3)
✅ Success!
```

**No action needed** - happens automatically!

---

### 7. 🎯 **Smart Query Detection**
**What it does:** Detects what you're asking about

**Query types:**
- 👤 **Person**: "ใคร", "who", "คน" → searches people
- 🏛️ **Organization**: "กระทรวง", "ministry" → searches orgs
- 🔗 **Relationship**: "รู้จัก", "connect" → finds paths
- 📋 **Position**: "ตำแหน่ง", "role" → searches positions
- 📅 **Timeline**: "เมื่อไหร่", "when" → time-based

**You'll see:**
```
🎯 Detected query type: person
🔗 Checking connection path between people...
```

---

## 🎓 Pro Tips

### Get Better Answers
1. **Be specific**: "รัฐมนตรีว่าการกระทรวงการคลัง" better than "รัฐมนตรี"
2. **Use Thai names**: "อนุทิน ชาญวีรกูล" better than "อนุทิน"
3. **Ask relationships**: "X connect to Y how?" gets path finding
4. **Use follow-ups**: Click suggested questions for deeper exploration

### Optimize Performance
1. **Enable streaming** for long answers (better perceived speed)
2. **Use caching** - repeat similar questions within 1 hour
3. **Check analytics** to see what works well

### Troubleshooting
1. **Rate limited?** Wait 10s or app will auto-retry
2. **Slow response?** Check if first time (no cache) or API is slow
3. **No answer?** Try rephrasing or check analytics for errors
4. **Wrong answer?** Use "🔄 Regenerate" button

---

## 📱 Quick Actions Reference

### Sidebar Buttons
- **+ New Chat**: Start fresh conversation
- **⚙️ Settings**: Toggle features, view analytics
- **Chat History**: Switch between conversations
- **🗑️ Delete**: Remove a conversation

### Message Actions
- **✏️ Edit**: Modify previous message
- **🔄 Regenerate**: Get new answer (bypasses cache)

### Settings Panel
- **🌊 Streaming responses**: Toggle streaming mode
- **Current model**: Shows which LLM is active
- **📊 Analytics**: View performance stats

---

## 🔥 Try These Example Queries

### Basic Queries
```
ใครคือนายกรัฐมนตรี?
รัฐมนตรีว่าการกระทรวงการคลัง?
ตำแหน่งของอนุทิน ชาญวีรกูล?
```

### Relationship Queries
```
อนุทินรู้จักใครบ้าง?
จุรินทร์เชื่อมกับอนุทินอย่างไร?
ใครทำงานในกระทรวงการคลัง?
```

### Aggregated Queries
```
รัฐมนตรีช่วยว่าการทั้งหมดมีใครบ้าง?
กระทรวงทั้งหมดมีอะไรบ้าง?
รัฐมนตรีแต่ละกระทรวง?
```

### Complex Queries
```
Stelligence network มีใครบ้าง?
รัฐมนตรีที่เกี่ยวข้องกับ Santisook?
อนุทินและจุรินทร์ทำงานด้วยกันที่ไหน?
```

---

## 🎨 UI Improvements (Already Applied)

- ✅ Bullet points on separate lines
- ✅ Full position names with ministry
- ✅ Grouped answers by category
- ✅ Clean formatting with headers
- ✅ Follow-up suggestions
- ✅ Real-time streaming (optional)

---

## ❓ FAQ

**Q: Why is the first query slow?**  
A: No cache yet. Subsequent queries are 10x faster.

**Q: How do I clear cache?**  
A: Wait 1 hour, or use "🔄 Regenerate" button.

**Q: Can I disable streaming?**  
A: Yes! Settings → Uncheck "🌊 Streaming responses"

**Q: Where are analytics stored?**  
A: `query_analytics.jsonl` (local file, not committed to git)

**Q: Does retry cost extra API credits?**  
A: No - only successful calls use credits.

**Q: What's the best model to use?**  
A: Currently using `deepseek/deepseek-chat` - free and good quality!

---

**🚀 Ready to explore? Start with "ใครคือนายกรัฐมนตรี?" and see the magic!**
