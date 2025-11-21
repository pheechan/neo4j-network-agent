# Connection Verification for Test Cases

Run these queries in your Streamlit app to verify actual connections:

## Quick Verification Query

Paste this into your Streamlit chat:

```
ตรวจสอบว่า วรวุฒิ หลายพูนสวัสดิ์ และ พิพัฒน์ รัชกิจประการ เชื่อมต่อกันหรือไม่
หาเส้นทางที่สั้นที่สุดระหว่างคน 2 คนนี้
```

## If that works, test all 10 scenarios from Test Case 4:

### 1. ฉัฐชัย มีชั้นช่วง → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "ฉัฐชัย มีชั้นช่วง" ไป "พิพัฒน์ รัชกิจประการ"
```

### 2. วรวุฒิ หลายพูนสวัสดิ์ → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "วรวุฒิ หลายพูนสวัสดิ์" ไป "พิพัฒน์ รัชกิจประการ"
```

### 3. ประมุข อุณหเลขกะ → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "ประมุข อุณหเลขกะ" ไป "พิพัฒน์ รัชกิจประการ"
```

### 4. ประเสริฐ สินสุขประเสริฐ → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "ประเสริฐ สินสุขประเสริฐ" ไป "พิพัฒน์ รัชกิจประการ"
```

### 5. รุ่งโรจน์กิติยศ - → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "รุ่งโรจน์กิติยศ" ไป "พิพัฒน์ รัชกิจประการ"
```

### 6. ปิติ นฤขัตรพิชัย → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "ปิติ นฤขัตรพิชัย" ไป "พิพัฒน์ รัชกิจประการ"
```

### 7. ชื่นสุมน นิวาทวงษ์ → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "ชื่นสุมน นิวาทวงษ์" ไป "พิพัฒน์ รัชกิจประการ"
```

### 8. สามารถ ถิระศักดิ์ → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "สามารถ ถิระศักดิ์" ไป "พิพัฒน์ รัชกิจประการ"
```

### 9. วรยุทธ อันเพียร → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "วรยุทธ อันเพียร" ไป "พิพัฒน์ รัชกิจประการ"
```

### 10. พูลพัฒน์ ลีสมบัติไพบูลย์ → พิพัฒน์ รัชกิจประการ
```
หาเส้นทางที่สั้นที่สุดจาก "พูลพัฒน์ ลีสมบัติไพบูลย์" ไป "พิพัฒน์ รัชกิจประการ"
```

---

## What to Check:

✅ **If connection exists**: You'll see the path with intermediate people
❌ **If no connection**: LLM will say "ไม่พบเส้นทางเชื่อมต่อ" or similar

## If many fail:

I'll need to find people who are ACTUALLY connected in your database. 

Can you:
1. Run the first query (วรวุฒิ → พิพัฒน์) in Streamlit
2. Tell me the result (connected or not connected)
3. If not connected, I'll query for people who ARE connected and update Test Case 4

---

## Alternative: Find Real Connections

If you want me to find real connected people, paste this query in Streamlit chat:

```
แสดง 10 คนที่มี connections มากที่สุด และบอกว่าแต่ละคนมีกี่ connections
```

Then I can use those highly-connected people to create working test cases!
