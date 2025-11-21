# Test Cases for Neo4j Network Agent

## Test Case 1: Optimal Path with Most Connected Intermediates
**Original request from boss:**
> เส้นทางที่สั้นที่สุด แต่ ระหว่างทางที่จะไปถึง Target มี connection มากที่สุด
> ระบุชื่อรัฐมนตรีที่ผ่าน network มากที่สุด

**Rewritten as clear prompt:**
```
หาเส้นทางที่สั้นที่สุดจาก [ชื่อต้นทาง] ไป [ชื่อเป้าหมาย] 
โดยเลือกเส้นทางที่ผ่านบุคคลที่มี connections มากที่สุด
และระบุว่าแต่ละคนมีกี่ connections
```

**Example queries:**

### Query 1: Find optimal path between two ministers
```
หาเส้นทางที่สั้นที่สุดจาก "อนุทิน ชาญวีรกูล" ไป "จุรินทร์ ลักษณวิศิษฏ์"
โดยเลือกเส้นทางที่ผ่านบุคคลที่มี connections มากที่สุด
ระบุชื่อเต็มและจำนวน connections ของแต่ละคนในเส้นทาง
```

**Expected output format:**
```
เส้นทางที่แนะนำ (ระยะทาง 2 ขั้น):

1. อนุทิน ชาญวีรกูล (ต้นทาง)
   - ตำแหน่ง: นายกรัฐมนตรี และรัฐมนตรีว่าการกระทรวงมหาดไทย
   - จำนวน connections: 15

2. พลเอก ประวิตร วงษ์สุวรรณ (คนกลาง - มี connections มากที่สุด)
   - ตำแหน่ง: รองนายกรัฐมนตรี
   - จำนวน connections: 28 ⭐ ← เลือกเส้นทางนี้เพราะมี connections มากที่สุด
   - เชื่อมต่อผ่าน: ประชุมคณะรัฐมนตรี

3. จุรินทร์ ลักษณวิศิษฏ์ (เป้าหมาย)
   - ตำแหน่ง: รองนายกรัฐมนตรี และรัฐมนตรีว่าการกระทรวงพาณิชย์
   - จำนวน connections: 12

**สรุป:**
✅ เส้นทางนี้มีระยะทาง 2 ขั้น (สั้นที่สุด)
✅ ผ่านคนกลางที่มี connections รวม 28 (มากที่สุด)
✅ โอกาสติดต่อสำเร็จสูงเพราะคนกลางมีอิทธิพลมาก
```

### Query 2: Find path through most connected ministers
```
จาก "Boss" (สมาชิกเครือข่าย Stelligence) ไป "พี่โด่ง" (รัฐมนตรี)
เส้นทางไหนผ่านรัฐมนตรีที่มี network มากที่สุด?
ระบุชื่อเต็ม ตำแหน่ง และจำนวน connections
```

### Query 3: Who has most connections to reach target
```
ถ้าอยากรู้จักกับ "อนุทิน ชาญวีรกูล" ควรรู้จักกับใครก่อน?
แนะนำคนที่มี connections มากที่สุด และสามารถเชื่อมต่อถึงอนุทินได้
```

---

## Test Case 2: Anti-Hallucination Test
**Purpose:** Ensure LLM doesn't say "ไม่มีข้อมูลกระทรวง" when ministry info exists in relationships

### Query 1: Person with ministry in relationships
```
อนุทิน ชาญวีรกูล ดำรงตำแหน่งอะไรบ้าง? ระบุกระทรวงด้วย
```

**Expected:** Should find ministry from relationships, NOT say "ไม่มีข้อมูลกระทรวง"

### Query 2: Multiple ministers
```
รัฐมนตรีว่าการแต่ละกระทรวง คือใครบ้าง?
ระบุชื่อเต็มและกระทรวงของแต่ละคน
```

**Expected:** Should list all ministers with their full ministry names

---

## Test Case 3: Network Analysis
### Query 1: Who is most connected in network
```
ในเครือข่าย Stelligence ใครมี connections มากที่สุด?
แสดง Top 5 พร้อมตำแหน่งและจำนวน connections
```

### Query 2: Ministry with most connections
```
กระทรวงใดมีสมาชิกที่เชื่อมต่อกับเครือข่ายภายนอกมากที่สุด?
```

---

## How to Use These Test Cases

1. **Copy the query** from above
2. **Paste into chat** in Streamlit app
3. **Check if output matches expectations:**
   - ✅ Ministry names are complete (not "ไม่มีข้อมูล")
   - ✅ Connection counts are shown
   - ✅ Optimal path is chosen (most connected intermediates)
   - ✅ No hallucination (only info from Context)

---

## Success Criteria

✅ **No Hallucination:** 
- LLM only uses info from Context
- Searches relationships thoroughly before saying "no data"

✅ **Optimal Path Selection:**
- Finds shortest path
- Among equal length paths, picks one with most connected intermediates
- Shows connection count for each person

✅ **Complete Information:**
- Full names with surnames
- Complete positions with ministry names
- Connection counts displayed clearly

---

## Notes for Testing

- Test with actual names from your Neo4j database
- Replace [ชื่อต้นทาง] and [ชื่อเป้าหมาย] with real names
- Check if Context shows relationships correctly
- Verify ministry information is found in relationships section
