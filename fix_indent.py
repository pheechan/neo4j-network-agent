#!/usr/bin/env python
# Fix indentation at line 1919

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 1919 - change from 5 tabs to 4 tabs
lines[1918] = '\t\t\t\tst.caption(f"⚠️ Added NO PATH warning to context")\n'

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed line 1919 indentation")
