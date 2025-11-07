# 🧪 Test Cases for Neo4j Network Agent

## Category 1: Person Information (บุคคล)

### Test 1.1: Single Position Query
**Query:** `อนุทิน ชาญวีรกูล ตำแหน่งอะไร`

**Expected Output:**
- ✅ Must show full ministry name: "รัฐมนตรีว่าการกระทรวงมหาดไทย"
- ✅ No preamble ("ตามข้อมูล...", "จาก Context...")
- ✅ Bullet points on separate lines
- ✅ Include role description

**What to Check:**
- [ ] Full ministry name shown?
- [ ] No preamble at start?
- [ ] Proper formatting (bullets, spacing)?
- [ ] Any additional context provided?

---

### Test 1.2: Multiple Positions
**Query:** `สุดารัตน์ เกยุราพันธุ์ ดำรงตำแหน่งอะไรบ้าง`

**Expected Output:**
- ✅ List ALL positions
- ✅ Each position with full ministry/agency name
- ✅ Bullet points on separate lines
- ✅ Brief role description for each

**What to Check:**
- [ ] All positions listed?
- [ ] Full names for each position?
- [ ] Good formatting?

---

### Test 1.3: Person Not Found
**Query:** `คนที่ชื่อ ทดสอบ ทดสอบ ตำแหน่งอะไร`

**Expected Output:**
- ✅ Clear "not found" message in Thai
- ✅ No hallucination (don't make up data)
- ✅ Suggest what user can do

**What to Check:**
- [ ] Honest "not found" response?
- [ ] No made-up information?

---

## Category 2: Position Queries (ตำแหน่ง)

### Test 2.1: Who Holds a Position
**Query:** `ใครคือนายกรัฐมนตรี`

**Expected Output:**
- ✅ Direct answer: "อนุทิน ชาญวีรกูล"
- ✅ Include full name + surname
- ✅ Add brief context about the role
- ✅ Mention other positions held (if any)

**What to Check:**
- [ ] Direct answer first?
- [ ] Full name shown?
- [ ] Additional context provided?

---

### Test 2.2: Multiple People in Same Position Type
**Query:** `ใครเป็นรองนายกรัฐมนตรีบ้าง`

**Expected Output:**
- ✅ Count: "มีทั้งหมด X ท่าน"
- ✅ List all with full names
- ✅ Each person on separate line with bullet
- ✅ Include their other responsibilities if any

**What to Check:**
- [ ] Count shown?
- [ ] All people listed?
- [ ] Clean formatting?

---

### Test 2.3: Ministers by Ministry
**Query:** `แต่ละรัฐมนตรีว่าการรับผิดชอบกระทรวงใดบ้าง`

**Expected Output:**
- ✅ Grouped by ministry
- ✅ Full "รัฐมนตรีว่าการกระทรวง[ชื่อ]" for each
- ✅ Count: "มีทั้งหมด X กระทรวง"
- ✅ Logical sorting (alphabetical or by importance)

**What to Check:**
- [ ] Grouped properly?
- [ ] Full ministry names?
- [ ] Count shown?
- [ ] No incomplete entries?

---

### Test 2.4: Deputy Ministers (รัฐมนตรีช่วยว่าการ)
**Query:** `มีรัฐมนตรีช่วยว่าการกี่คน และรับผิดชอบกระทรวงใด`

**Expected Output:**
- ✅ Total count first
- ✅ Grouped by ministry
- ✅ Each person with full "รัฐมนตรีช่วยว่าการกระทรวง[ชื่อ]"
- ✅ Clean categorization

**What to Check:**
- [ ] Count correct and shown first?
- [ ] Grouped by ministry?
- [ ] Full position names?

---

## Category 3: Relationship Queries (ความสัมพันธ์)

### Test 3.1: Stelligence Network - Santisook
**Query:** `Santisook มีความสัมพันธ์กับใครบ้าง`

**Expected Output:**
- ✅ Show network summary: "🌐 Santisook Network: X คน"
- ✅ List ALL people with Stelligence: Santisook
- ✅ Include their positions and ministries
- ✅ Should show 30-50+ people (not just 2-3)

**What to Check:**
- [ ] Network count shown at top?
- [ ] Large number of people (30+)?
- [ ] All with proper positions?
- [ ] Clean grouping?

---

### Test 3.2: Other Networks
**Query:** `Por รู้จักกับใครบ้าง`

**Expected Output:**
- ✅ Por network summary
- ✅ All network members
- ✅ Positions and organizations

**What to Check:**
- [ ] Por network complete?
- [ ] Same quality as Santisook test?

---

### Test 3.3: Connect By Relationships
**Query:** `นเรศ ธำรงค์ทิพยคุณ เชื่อมโยงกับใคร`

**Expected Output:**
- ✅ Show all "Connect by" relationships
- ✅ Show direct relationships (colleagues, etc.)
- ✅ Organized by type (people, positions, agencies)

**What to Check:**
- [ ] Multiple relationship types shown?
- [ ] Organized clearly?

---

## Category 4: Organization Queries (องค์กร)

### Test 4.1: Ministry Information
**Query:** `กระทรวงมหาดไทยมีหน้าที่อะไร`

**Expected Output:**
- ✅ Ministry responsibilities
- ✅ Key people (minister, deputies)
- ✅ Related agencies if any

**What to Check:**
- [ ] Relevant information shown?
- [ ] People with full positions?

---

### Test 4.2: People in an Agency
**Query:** `มีใครบ้างในจุฬาลงกรณ์มหาวิทยาลัย`

**Expected Output:**
- ✅ List all people from that agency
- ✅ Their positions
- ✅ Count

**What to Check:**
- [ ] Complete list?
- [ ] Positions shown?

---

## Category 5: Aggregation Queries (รวบรวม)

### Test 5.1: Count Positions
**Query:** `มีรัฐมนตรีกี่คนทั้งหมด`

**Expected Output:**
- ✅ Clear count
- ✅ Breakdown by type (ว่าการ vs ช่วยว่าการ)
- ✅ Optional: List names

**What to Check:**
- [ ] Accurate count?
- [ ] Breakdown shown?

---

### Test 5.2: All Ministries
**Query:** `มีกระทรวงอะไรบ้าง`

**Expected Output:**
- ✅ Count first
- ✅ List all ministries
- ✅ Each with minister name
- ✅ Alphabetical or logical order

**What to Check:**
- [ ] Complete list?
- [ ] Count shown?
- [ ] Ministers included?

---

### Test 5.3: Complex Aggregation
**Query:** `แต่ละกระทรวงมีรัฐมนตรีว่าการและรัฐมนตรีช่วยว่าการเป็นใครบ้าง`

**Expected Output:**
- ✅ Grouped by ministry
- ✅ Show minister (ว่าการ) and deputies (ช่วยว่าการ)
- ✅ Full position names
- ✅ Clear hierarchy

**What to Check:**
- [ ] All ministries covered?
- [ ] Clear hierarchy shown?
- [ ] Full position names?

---

## Category 6: Edge Cases (กรณีพิเศษ)

### Test 6.1: Ambiguous Name
**Query:** `อนุทิน ตำแหน่งอะไร` (ไม่มีนามสกุล)

**Expected Output:**
- ✅ Should find "อนุทิน ชาญวีรกูล"
- ✅ Show full name in response
- ✅ Complete position info

**What to Check:**
- [ ] Found correct person?
- [ ] Full name shown?

---

### Test 6.2: English Query
**Query:** `Who is the Prime Minister?`

**Expected Output:**
- ✅ Answer in English
- ✅ Same quality as Thai responses
- ✅ Full names and titles

**What to Check:**
- [ ] English response?
- [ ] Same quality?

---

### Test 6.3: Mixed Thai-English
**Query:** `Santisook เป็นใคร`

**Expected Output:**
- ✅ Handle mixed language
- ✅ Thai response (follow query language)
- ✅ Complete info

**What to Check:**
- [ ] Handled correctly?

---

### Test 6.4: Very Short Query
**Query:** `นายกฯ`

**Expected Output:**
- ✅ Understand abbreviation
- ✅ Provide full answer
- ✅ Explain abbreviation

**What to Check:**
- [ ] Understood abbreviation?

---

## Category 7: Data Quality Checks (ตรวจสอบคุณภาพ)

### Test 7.1: No Hallucination
**Query:** `รัฐมนตรีว่าการกระทรวงอวกาศคือใคร` (ไม่มีกระทรวงนี้)

**Expected Output:**
- ✅ Honest "not found" or "ไม่มีข้อมูล"
- ✅ NO made-up information
- ✅ Suggest related info or correction

**What to Check:**
- [ ] No fake data?
- [ ] Honest response?

---

### Test 7.2: Incomplete Data Handling
**Query:** `[Person with partial data] ทำงานที่ไหน`

**Expected Output:**
- ✅ Show what data is available
- ✅ Acknowledge what's missing
- ✅ Don't make up missing info

**What to Check:**
- [ ] Only shows available data?
- [ ] Acknowledges gaps?

---

### Test 7.3: Relationship Without Ministry
**Query:** `[Person without ministry] ตำแหน่งอะไร`

**Expected Output:**
- ✅ Show position
- ✅ Show organization/agency if available
- ✅ Don't force ministry name if not available

**What to Check:**
- [ ] Handles missing ministry gracefully?
- [ ] Shows alternative org info?

---

## Category 8: Format & Style Checks (รูปแบบ)

### Test 8.1: Preamble Check
**Run ANY query and check:**
- [ ] ❌ Does NOT start with: "ตามข้อมูล...", "จาก Context...", "ตาม Knowledge Graph..."
- [ ] ✅ Starts directly with answer

---

### Test 8.2: Bullet Point Formatting
**Run list queries and check:**
- [ ] ✅ Each bullet on new line
- [ ] ✅ NOT: "มี 3 คน คนที่ 1... คนที่ 2... คนที่ 3..."
- [ ] ✅ YES: Line breaks between items

---

### Test 8.3: Ministry Name Completeness
**Run position queries and check:**
- [ ] ✅ "รัฐมนตรีว่าการกระทรวง[ชื่อ]"
- [ ] ❌ NOT just "รัฐมนตรีว่าการ"

---

### Test 8.4: Suggested Follow-ups
**Any query - check if includes:**
- [ ] ✅ "คุณอาจสนใจ:" or similar
- [ ] ✅ Relevant follow-up questions
- [ ] ✅ Questions actually related to topic

---

## Quick Test Checklist ✓

For **EVERY** test, verify:
1. [ ] No preamble ("ตามข้อมูล...", etc.)
2. [ ] Full ministry names for all positions
3. [ ] Bullet points on separate lines
4. [ ] Full names (ชื่อ-นามสกุล)
5. [ ] Synthesized/grouped data (not raw dump)
6. [ ] Main answer first, then details
7. [ ] Suggested follow-up questions at end
8. [ ] No hallucinated data

---

## Priority Test Sequence

**Start with these critical tests:**
1. Test 1.1 (อนุทิน position) - Tests CRITICAL RULE #1
2. Test 2.3 (Ministers by ministry) - Tests aggregation + full names
3. Test 3.1 (Santisook network) - Tests hybrid search completeness
4. Test 5.3 (Complex aggregation) - Tests data synthesis
5. Test 6.1 (Ambiguous name) - Tests search quality
6. Test 7.1 (Non-existent ministry) - Tests hallucination prevention

---

## Scoring System

**Give each test a score:**
- ✅ **5/5** - Perfect (all criteria met)
- ⚠️ **3-4/5** - Good (minor issues)
- ❌ **1-2/5** - Poor (major issues)
- 💥 **0/5** - Failed (completely wrong)

**Target:** Average score ≥ 4.0/5.0

---

## Report Format

```
Test X.X: [Name]
Query: "[query]"
Score: X/5
Issues:
- [ ] Issue 1
- [ ] Issue 2
What worked well:
- [X] Feature 1
- [X] Feature 2
```
